from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.api.message_components import Node, Plain, Image, Nodes, Reply, BaseMessageComponent
import asyncio
from io import BytesIO
import time
import os
import random
from google import genai
from PIL import Image as PILImage
from google.genai.types import HttpOptions
from astrbot.core.utils.io import download_file
import functools
from typing import List, Optional, Dict, Tuple, AsyncGenerator, Any
from openai import OpenAI
from collections import deque
import base64
import json
from pathlib import Path
import re



@register("gemini_artist_plugin_copy", "nichinichisou", "基于 Google Gemini 和 OpenRouter 格式 API 的AI绘画插件", "1.5.0")
class GeminiArtist(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        self.config = config
        
        # API 类型配置："Google" 或 "OpenRouter" 
        self.api_type = config.get("api_type", "Google")
        api_key_list_from_config = config.get("api_key", [])
        self.api_base_url_from_config = config.get("api_base_url", "https://generativelanguage.googleapis.com")
        self.model_name_from_config = config.get("model", "gemini-2.0-flash-exp")
        self.group_whitelist = config.get("group_whitelist", [])
        self.robot_id_from_config = config.get("robot_self_id") 
        self.random_api_key_selection = config.get("random_api_key_selection", False)
        self.enable_base_reference_image = config.get("enable_base_reference_image", False)
        self.base_reference_image_path = config.get("base_reference_image_path", "")
        # 存储正在等待输入的用户，键为 (user_id, group_id)
        self.waiting_users = {}  # {(user_id, group_id): expiry_time}
        # 存储用户收集到的文本和图片，键为 (user_id, group_id)
        self.user_inputs = {} # {(user_id, group_id): {'messages': [{'text': '', 'images': [], 'timestamp': float}]}}
        self.wait_time_from_config = config.get("wait_time", 30)

        # 存储用户发送的图片URL缓存
        self.image_history_cache: Dict[Tuple[str, str], deque[Tuple[str, Optional[str]]]] = {}
        self.max_cached_images = self.config.get("max_cached_images", 5)

        # 设置插件的临时文件目录
        shared_data_path = Path(__file__).resolve().parent.parent.parent
        self.plugin_temp_base_dir = os.path.join(shared_data_path, "gemini_artist_temp")
        os.makedirs(self.plugin_temp_base_dir, exist_ok=True)
        self.temp_dir = self.plugin_temp_base_dir

        self.enable_hinting = self.config.get("enable_hinting", True)

        self.api_keys = [
            key.strip()
            for key in api_key_list_from_config
            if isinstance(key, str) and key.strip()
        ]
        self.current_api_key_index = 0

        if not self.api_keys:
            logger.warning("Gemini API密钥未配置或配置为空。插件可能无法正常工作。")

        # 配置临时文件清理任务
        self.cleanup_interval_seconds = self.config.get("temp_cleanup_interval_seconds", 3600 * 6)
        self.cleanup_older_than_seconds = self.config.get("temp_cleanup_files_older_than_seconds", 86400 * 3)
        self._background_cleanup_task = None

        # 启动后台定时清理任务
        if self.cleanup_interval_seconds > 0:
            self._background_cleanup_task = asyncio.create_task(self._periodic_temp_dir_cleanup())
            logger.info(f"GeminiArtist: 已启动定时清理任务，每隔 {self.cleanup_interval_seconds} 秒清理临时目录 {self.temp_dir} 中超过 {self.cleanup_older_than_seconds} 秒的文件。")
        else:
            logger.info("GeminiArtist: 定时清理功能已禁用 (temp_cleanup_interval_seconds <= 0)。")

    def _blocking_cleanup_temp_dir_logic(self, older_than_seconds: int) -> Tuple[int, int]:
        """
        同步执行临时目录清理的逻辑，移除旧文件。
        """
        if not os.path.isdir(self.temp_dir):
            return 0, 0
        now, cleaned_count, error_count = time.time(), 0, 0
        try:
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        if (now - os.path.getmtime(file_path)) > older_than_seconds:
                            os.remove(file_path)
                            cleaned_count += 1
                except Exception as e_file:
                    logger.error(f"清理临时文件 {file_path} 时出错: {e_file}")
                    error_count += 1
        except Exception as e_list:
            logger.error(f"列出目录 {self.temp_dir} 进行清理时出错: {e_list}")
            error_count += 1
        if cleaned_count > 0 or error_count > 0:
            logger.info(f"临时目录清理: 移除 {cleaned_count} 文件, 发生 {error_count} 错误 @ {self.temp_dir}")
        return cleaned_count, error_count

    async def _periodic_temp_dir_cleanup(self):
        """
        周期性地清理临时目录的后台任务。
        """
        while True:
            await asyncio.sleep(self.cleanup_interval_seconds)
            logger.info(f"定时清理触发: {self.temp_dir}")
            try:
                cleanup_func = functools.partial(self._blocking_cleanup_temp_dir_logic, self.cleanup_older_than_seconds)
                await asyncio.to_thread(cleanup_func)
            except asyncio.CancelledError:
                logger.info("定时清理任务已取消。")
                break
            except Exception as e:
                logger.error(f"定时清理任务出错: {e}", exc_info=True)

    def store_user_image(self, user_id: str, group_id: str, image_url: str, original_filename: Optional[str] = None) -> None:
        """
        将用户发送的图片URL存储到缓存中。
        """
        key = (user_id, group_id)
        if key not in self.image_history_cache:
            self.image_history_cache[key] = deque(maxlen=self.max_cached_images)
        self.image_history_cache[key].append((image_url, original_filename))
        logger.debug(f"已存储用户 {user_id} group_id {group_id} 图片URL: {image_url} (缓存 {len(self.image_history_cache[key])}/{self.max_cached_images})")

    async def download_pil_image_from_url(self, image_url: str, context_description: str = "图片") -> Optional[PILImage.Image]:
        """
        从给定的URL下载图片并返回PIL Image对象。
        """
        logger.info(f"尝试使用 astrbot.core.utils.io.download_file 下载 {context_description} URL: {image_url}")

        # 尝试从URL中获取文件扩展名
        ext = ".png"
        try:
            path_part = image_url.split('?')[0].split('#')[0]
            base_name = os.path.basename(path_part)
            _, url_ext = os.path.splitext(base_name)
            if url_ext and url_ext.startswith('.') and len(url_ext) <= 5:
                ext = url_ext.lower()
            elif image_url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                for known_ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                    if image_url.lower().endswith(known_ext):
                        ext = known_ext
                        break
        except Exception as e_ext:
            logger.debug(f"从URL {image_url} 获取扩展名时出错: {e_ext}，使用默认扩展名 {ext}")

        filename = f"gemini_artist_temp_{time.time()}_{random.randint(1000,9999)}{ext}"
        target_file_path = os.path.join(self.temp_dir, filename)

        os.makedirs(self.temp_dir, exist_ok=True)

        try:
            await download_file(url=image_url, path=target_file_path, show_progress=False)

            if os.path.exists(target_file_path) and os.path.isfile(target_file_path) and os.path.getsize(target_file_path) > 0:
                img_pil = PILImage.open(target_file_path)
                img_pil.load()
                if img_pil.mode != 'RGBA':
                    img_pil = img_pil.convert('RGBA')
                logger.info(f"图片从 {img_pil.mode} 转换为 RGBA 模式: {target_file_path}")


                logger.info(f"成功使用 download_file 下载并加载 {context_description} 从 {image_url} (本地文件: {target_file_path})")
                return img_pil
            else:
                logger.error(f"download_file 声称完成，但在路径 '{target_file_path}' 未找到有效文件或文件为空。URL: {image_url}")
                if os.path.exists(target_file_path):
                    try:
                        os.remove(target_file_path)
                    except Exception:
                        pass
                return None

        except FileNotFoundError:
            logger.error(f"尝试写入下载文件时发生 FileNotFoundError，请检查临时目录 '{self.temp_dir}' 是否有效且可写。 URL: {image_url}", exc_info=True)
            return None
        except PILImage.UnidentifiedImageError:
            logger.error(f"Pillow无法识别下载的图片文件 {target_file_path}。可能不是有效的图片格式或文件已损坏。 URL: {image_url}", exc_info=True)
            if os.path.exists(target_file_path):
                try:
                    os.remove(target_file_path)
                except Exception as e_rem:
                    logger.warning(f"清理无效下载文件 {target_file_path} 失败: {e_rem}")
            return None
        except Exception as e:
            logger.error(f"调用 download_file(url='{image_url}', path='{target_file_path}') 时发生错误: {type(e).__name__} - {e}", exc_info=True)
            if os.path.exists(target_file_path) and os.path.getsize(target_file_path) == 0:
                try:
                    os.remove(target_file_path)
                except Exception:
                    pass
            return None

    def _load_base_reference_image(self) -> Optional[PILImage.Image]:
        """
        从配置的路径加载默认的基础参考图。
        """
        if not self.base_reference_image_path:
            return None

        # 将路径相对于 AstrBot 根目录解析
        astrbot_root = Path(__file__).resolve().parent.parent.parent.parent
        image_path = Path(self.base_reference_image_path)
        if not image_path.is_absolute():
            image_path = astrbot_root / image_path

        if image_path.exists() and image_path.is_file():
            try:
                logger.info(f"正在加载默认参考图: {image_path}")
                img_pil = PILImage.open(image_path)
                img_pil.load()  # 确保图片数据已加载
                # 转换为RGBA以获得最佳兼容性
                return img_pil.convert('RGBA') if img_pil.mode != 'RGBA' else img_pil
            except Exception as e:
                logger.error(f"加载默认参考图失败: {image_path}, 错误: {e}")
                return None
        else:
            logger.warning(f"配置的默认参考图路径不存在或不是一个文件: {image_path}")
            return None

    async def get_user_recent_image_pil_from_cache(self, user_id: str, group_id: str, index: int = 1) -> Optional[PILImage.Image]:
        """
        从用户图片缓存中获取指定索引的图片并下载为PIL Image对象。
        """
        key = (user_id, group_id)
        if key not in self.image_history_cache or not self.image_history_cache[key]:
            logger.debug(f"缓存中未找到用户 {user_id} group_id {group_id} 的图片URL。")
            return None
        cached_items = list(self.image_history_cache[key])
        if not (0 < index <= len(cached_items)):
            logger.debug(f"请求的图片URL索引 {index} 超出用户 {user_id} group_id {group_id} 缓存范围 ({len(cached_items)} 条)。")
            return None
        image_ref_str, _ = cached_items[-index]
        if image_ref_str.startswith("data:image"):
            logger.info(f"从缓存加载Base64 Data URL (用户 {user_id}, 上下文 {group_id}, 索引 {index})")
            try:
                header, encoded = image_ref_str.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                pil_image = PILImage.open(BytesIO(image_bytes))
                return pil_image.convert('RGBA') if pil_image.mode != 'RGBA' else pil_image
            except Exception as e:
                logger.error(f"从缓存的Data URL解码图片失败: {e}")
                return None
        elif image_ref_str.startswith("http://") or image_ref_str.startswith("https://"):
            logger.info(f"从缓存加载HTTP URL并下载 (用户 {user_id}, 上下文 {group_id}, 索引 {index}): {image_ref_str}")
            return await self.download_pil_image_from_url(image_ref_str, f"缓存图片 (HTTP)")
        elif os.path.exists(image_ref_str): # 假设是本地文件路径
             logger.info(f"从缓存加载本地文件路径 (用户 {user_id}, 上下文 {group_id}, 索引 {index}): {image_ref_str}")
             try:
                pil_image = PILImage.open(image_ref_str)
                return pil_image.convert('RGBA') if pil_image.mode != 'RGBA' else pil_image
             except Exception as e:
                logger.error(f"从缓存的本地路径加载图片失败: {e}")
                return None
        else:
            logger.warning(f"缓存中的图片引用格式未知或无效: {image_ref_str[:100]}...")
            return None

    @filter.event_message_type(EventMessageType.ALL)
    async def cache_user_images(self, event: AstrMessageEvent):
        """
        监听所有消息，将用户发送的图片URL缓存起来。
        """
        if not hasattr(event, 'message_obj') or not hasattr(event.message_obj, 'type'):
            return
        user_id = event.get_sender_id()
        if self.robot_id_from_config and user_id == self.robot_id_from_config:
            return
        group_id = event.message_obj.group_id
        if self.group_whitelist:
            identifier_to_check = event.message_obj.group_id if event.message_obj.group_id else user_id
            if str(identifier_to_check) not in [str(whitelisted_id) for whitelisted_id in self.group_whitelist]:
                return
        if group_id == "":
            group_id = user_id
            logger.debug(f"收到来自用户 {user_id} group_id {group_id} 的消息。")
        for msg_component in event.get_messages():
            if isinstance(msg_component, Image) and hasattr(msg_component, 'url') and msg_component.url:
                self.store_user_image(user_id, group_id, msg_component.url, getattr(msg_component, 'file', None))

    @filter.llm_tool(name="gemini_draw")
    async def gemini_draw(self, event: AstrMessageEvent, prompt: str, image_index: int = 0, reference_bot: bool = False) -> AsyncGenerator[Any, None]:
        '''
        AI图像生成与编辑工具。支持文生图、图生图、图像编辑等多种功能。
        Args:
            prompt (string): 图像生成或编辑的详细描述，当用户使用"再画一张"、"重新生成一张"时使用之前绘画的提示词。
            image_index (number, optional): 引用历史图片数量。0=不引用，1=引用最新1张，2=引用最新2张，依此类推。默认为0。
            reference_bot (boolean, optional): 是否引用机器人之前生成的图片。True=引用机器人生成的，False=引用用户发送的。默认为False。
        '''
        if not self.api_keys:
            yield event.plain_result("请联系管理员配置Gemini API密钥。")
            return
        if not hasattr(event, 'message_obj') or not hasattr(event.message_obj, 'type'):
            logger.error(f"gemini_draw: 事件对象缺少 message_obj 或 type 属性。")
            yield event.plain_result("处理消息时出错。")
            return

        command_sender_id = event.get_sender_id()
        group_id = event.message_obj.group_id

        if self.group_whitelist and str(event.message_obj.group_id or command_sender_id) not in map(str, self.group_whitelist):
            return
        if self.robot_id_from_config and command_sender_id == self.robot_id_from_config:
            return

        all_text = prompt.strip()
        all_images_pil: List[PILImage.Image] = []
        used_default_image = False # 新增：标记是否使用了默认参考图

        # 优先处理回复消息中的图片
        replied_image_pil: Optional[PILImage.Image] = None
        message_chain = event.get_messages()

        for msg_component in message_chain:
            if isinstance(msg_component, Reply):
                logger.debug(f"检测到回复消息。尝试解析被引用的图片。Reply component dir: {dir(msg_component)}")
                if hasattr(msg_component, '__dict__'):
                    logger.debug(f"Reply component vars: {vars(msg_component)}")

                source_chain: Optional[List[BaseMessageComponent]] = None
                # 尝试从回复消息中获取图片链
                if hasattr(msg_component, 'chain') and isinstance(msg_component.chain, list):
                    source_chain = msg_component.chain
                    logger.debug("Reply component has 'chain' attribute.")
                elif hasattr(msg_component, 'message') and isinstance(msg_component.message, list):
                    source_chain = msg_component.message
                    logger.debug("Reply component has 'message' attribute (list).")
                elif hasattr(msg_component, 'source') and hasattr(msg_component.source, 'message_chain') and isinstance(msg_component.source.message_chain, list):
                    source_chain = msg_component.source.message_chain
                    logger.debug("Reply component has 'source.message_chain' attribute.")

                if source_chain:
                    for replied_part in source_chain:
                        if isinstance(replied_part, Image) and hasattr(replied_part, 'url') and replied_part.url:
                            replied_image_pil = await self.download_pil_image_from_url(replied_part.url, "直接引用的消息中的图片")
                            if replied_image_pil:
                                logger.info("成功从直接引用的消息中加载了图片作为参考。")
                                all_images_pil.append(replied_image_pil)
                if replied_image_pil:
                    logger.info("使用直接引用的图片作为唯一参考，忽略 image_index 和 reference_user_id。")
                    image_index = 0
                    reference_bot = False
                    break
                break

        # 如果没有直接引用的图片，且指定了图片索引，则尝试从缓存中获取
        if not all_images_pil and image_index > 0:
            num_images_to_fetch = image_index
            if reference_bot == True:
                user_id_for_cache_lookup = self.robot_id_from_config
            else:
                user_id_for_cache_lookup = command_sender_id
            group_id_for_cache_lookup = event.message_obj.group_id or command_sender_id
            logger.info(f"尝试从用户 {user_id_for_cache_lookup} (上下文 {group_id_for_cache_lookup}) 缓存获取最新的 {num_images_to_fetch} 张图片。")

            key_for_cache = (user_id_for_cache_lookup, group_id_for_cache_lookup)
            if key_for_cache not in self.image_history_cache or not self.image_history_cache[key_for_cache]:
                message = f"缓存中未找到用户 {user_id_for_cache_lookup} (上下文 {group_id_for_cache_lookup}) 的图片历史。"
                logger.warning(message)
                # 直接跳过获取缓存图片的逻辑
            else:
                cached_items_list = list(self.image_history_cache[key_for_cache])
                # 确保不要请求超过缓存数量的图片
                actual_num_to_fetch = min(num_images_to_fetch, len(cached_items_list))

                if actual_num_to_fetch == 0:
                    message = f"用户 {user_id_for_cache_lookup} (上下文 {group_id_for_cache_lookup}) 的图片历史为空，无法按数量 {num_images_to_fetch} 获取参考图。"
                    logger.warning(message)

                fetched_count = 0
                # 从最新的开始获取 (倒数第1, 倒数第2, ..., 倒数第 actual_num_to_fetch)
                for i in range(1, actual_num_to_fetch + 1):
                    pil_image_from_cache = await self.get_user_recent_image_pil_from_cache(
                        user_id_for_cache_lookup,
                        group_id_for_cache_lookup,
                        i 
                    )
                    if pil_image_from_cache:
                        all_images_pil.append(pil_image_from_cache)
                        fetched_count +=1
                    else:
                        logger.warning(f"未能加载用户 {user_id_for_cache_lookup} (上下文 {group_id_for_cache_lookup}) 的倒数第 {i} 张图片。")
                
                if fetched_count == 0 and num_images_to_fetch > 0 : # 如果指定要图但一张都没取到
                    message = f"尝试获取最新的 {num_images_to_fetch} 张图片，但未能成功加载任何一张。"
                    logger.warning(message)
                
                logger.info(f"成功从缓存加载了 {fetched_count} 张参考图片。")

        # 如果没有任何用户提供的参考图，则尝试加载默认参考图
        if not all_images_pil and self.enable_base_reference_image:
            base_image = self._load_base_reference_image()
            if base_image:
                all_images_pil.append(base_image)
                used_default_image = True # 设置标记
                logger.info("已使用默认参考图。")

        if not all_text and not all_images_pil:
            yield event.plain_result("请提供文本描述，或通过回复图片/指定图片索引及可选的参考用户来提供有效的参考图片。")
            event.stop_event()
            return

        # 在提示词前添加英文前缀,提高调用绘画成功率
        if all_text:
            all_text = f"Generate/modify images using the following prompt: {all_text}"

        if self.enable_hinting:
            yield event.plain_result("正在生成图片，请稍候...")

        try:
            logger.debug(f"gemini_draw: 调用 API 生成 (API类型: {self.api_type}, 文本: '{all_text[:100]}...', PIL图片数: {len(all_images_pil)})")
            
            # 根据API类型调用相应的生成方法
            if self.api_type == "OpenRouter":
                result = await self.openrouter_generate(all_text, all_images_pil)
            else:
                # 默认使用 Google Gemini API
                result = await self.gemini_generate(all_text, all_images_pil)
            
            logger.debug(f"gemini_draw: API 调用完成。")

            if result is None or not isinstance(result, dict):
                logger.error(f"gemini_draw: gemini_generate 返回无效结果: {type(result)}")
                yield event.plain_result("处理图片时发生内部错误。")
                event.stop_event()
                return

            text_response = result.get('text', '').strip()
            image_paths = result.get('image_paths', [])
            logger.debug(f"gemini_generate 返回: 文本预览='{text_response[:100]}...', 生成图片数={len(image_paths)}")
            if image_paths:
                logger.info(f"准备缓存 {len(image_paths)} 张Gemini生成的图片路径...")
            for i, img_path in enumerate(image_paths):
                if os.path.exists(img_path):
                    # 缓存本地文件路径。command_sender_id 是触发这次生成的用户。
                    # current_session_context_id 是图片生成的上下文（群或私聊）。
                    self.store_user_image(
                        command_sender_id, # 图片归属于触发操作的用户
                        group_id, # 在当前会话上下文中
                        img_path, # 缓存的是本地文件路径
                        f"gemini_generated_{i+1}_{os.path.basename(img_path)}"
                    )
                else:
                    logger.warning(f"Gemini生成的图片路径无效，无法缓存或发送: {img_path}")
            if not text_response and not image_paths:
                logger.warning("gemini_draw: API未返回任何文本或生成的图片内容。")
                yield event.plain_result("未能从API获取任何内容。")
                event.stop_event()
                return

            # 如果只有一张图片或没有图片，则直接发送
            if len(image_paths) < 2:
                chain = []
                # if text_response:
                #     chain.append(Plain(text_response))
                for img_path in image_paths:
                    if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        chain.append(Image.fromFileSystem(img_path))
                if chain:
                    # 构建给LLM的反馈信息
                    llm_feedback = f"你生成了 {len(image_paths)} 张图片。"
                    if used_default_image:
                        llm_feedback += "由于用户没有提供参考图，你使用了插件预设的默认参考图进行创作。"
                    llm_feedback += ("这些图片已经发送给用户，并且用户可以通过图片索引（例如，image_index=1 代表最新生成的这张/这些图片）来引用它们进行后续操作。"
                                     "请根据用户的原始意图和这些新生成的图文内容继续对话。")

                    tool_output_data = {
                        "generated_text": text_response,
                        "number_of_images_generated": len(image_paths),
                        "user_instruction_for_llm": llm_feedback
                        }
                            # 工具的返回值应该是这个JSON字符串
                            # 当LLM调用此工具后，这个字符串会作为工具结果进入LLM的上下文
                    yield json.dumps(tool_output_data, ensure_ascii=False)
                    yield event.chain_result(chain)
                else:
                    if text_response:
                        yield event.plain_result(text_response)
                    else:
                        yield event.plain_result("抱歉，未能生成有效内容。")
                return
            else:
            # 如果有多张图片，尝试以合并转发消息的形式发送
                bot_id_for_node_str = event.message_obj.self_id or self.robot_id_from_config or self.config.get("bot_id")
                bot_id_for_node = int(str(bot_id_for_node_str).strip()) if bot_id_for_node_str and str(bot_id_for_node_str).strip().isdigit() else None
                if bot_id_for_node is None:
                    logger.error(f"gemini_draw: 无法确定有效的 bot_id。尝试普通发送。")
                    chain = []
                    if text_response:
                        chain.append(Plain(text_response))
                    for img_path in image_paths:
                        if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            chain.append(Image.fromFileSystem(img_path))
                    if chain:
                        yield event.chain_result(chain)
                    else:
                        yield event.plain_result("抱歉，未能生成有效内容。")
                    return

                bot_name_for_node = str(self.config.get("bot_name", "绘图助手")).strip() or "绘图助手"
                ns = Nodes([])
                paragraphs = []  # 初始化为空列表
                if text_response:
                    paragraphs = text_response.split('\n\n')
                if paragraphs:
                    ns.nodes.append(Node(
                            user_id=bot_id_for_node,nickname=bot_name_for_node,content=[Plain(paragraphs[0])]
                        ))
                for idx, img_path in enumerate(image_paths): 
                    if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        if len(paragraphs) <= 1:
                            content = [Plain(""), Image.fromFileSystem(img_path)]
                        elif idx + 1 < len(paragraphs):
                            content = [Plain(paragraphs[idx+1]), Image.fromFileSystem(img_path)]
                        else:
                            content = [Plain(""), Image.fromFileSystem(img_path)]
                        
                        ns.nodes.append(Node(
                            user_id=bot_id_for_node,
                            nickname=bot_name_for_node,
                            content=content
                        ))
                if ns:
                    # 构建给LLM的反馈信息
                    llm_feedback = f"你生成了 {len(image_paths)} 张图片。"
                    if used_default_image:
                        llm_feedback += "由于用户没有提供参考图，你使用了插件预设的默认参考图进行创作。"
                    llm_feedback += ("这些图片已经发送给用户，并且用户可以通过图片索引（例如，image_index=1 代表最新生成的这张/这些图片）来引用它们进行后续操作。"
                                     "请根据用户的原始意图和这些新生成的图文内容继续对话。")

                    tool_output_data = {
                        "generated_text": text_response,
                        "number_of_images_generated": len(image_paths),
                        "user_instruction_for_llm": llm_feedback
                    }
                    yield json.dumps(tool_output_data, ensure_ascii=False)
                    yield event.chain_result([ns])
                else:
                    yield event.plain_result("抱歉，未能生成有效内容。")

        except Exception as e:
            logger.error(f"gemini_draw 未知错误: {e}", exc_info=True)
            yield event.plain_result(f"处理请求时发生意外错误: {str(e)}")
    @filter.command("draw")
    async def initiate_creation_session(self, event: AstrMessageEvent):
        """处理 /draw 命令，启动绘图会话。(旧版功能)"""
        if not self.api_keys:
            yield event.plain_result("请联系管理员配置Gemini API密钥 (api_keys)")
            return

        if not hasattr(event, 'message_obj') or not hasattr(event.message_obj, 'type'):
             logger.error(f"initiate_creation_session: 事件对象缺少 message_obj 或 type 属性: {type(event)}")
             yield event.plain_result("处理消息类型时出错，请联系管理员。")
             return

        user_id = event.get_sender_id()
        user_name = event.get_sender_name()
        
        group_id = user_id 
        is_group_message = hasattr(event.message_obj, 'group_id') and event.message_obj.group_id is not None and event.message_obj.group_id != ""


        if is_group_message:
            group_id = event.message_obj.group_id
        
        if self.group_whitelist:
            identifier_to_check = group_id if is_group_message else user_id
            if str(identifier_to_check) not in [str(whitelisted_id) for whitelisted_id in self.group_whitelist]:
                logger.info(f"initiate_creation_session: 用户/群组 {identifier_to_check} 不在白名单中，已忽略 /draw 命令。")
                return # No reply for non-whitelisted

        session_key = (user_id, str(group_id)) # Ensure group_id is string for key consistency

        if session_key in self.waiting_users and time.time() < self.waiting_users[session_key]: # Check expiry
             expiry_time = self.waiting_users[session_key]
             remaining_time = int(expiry_time - time.time())
             yield event.plain_result(f"您已经在当前会话有一个正在进行的绘制任务，请先完成或等待超时 ({remaining_time}秒后)。")
             return
        elif session_key in self.waiting_users: # Expired entry
            del self.waiting_users[session_key]
            if session_key in self.user_inputs:
                del self.user_inputs[session_key]


        self.waiting_users[session_key] = time.time() + self.wait_time_from_config
        self.user_inputs[session_key] = {'messages': []}
        
        logger.debug(f"Gemini_Draw (Command): User {user_id} started draw. Session ID: {group_id}, Session Key: {session_key}. Waiting state set.")
        yield event.plain_result(f"好的 {user_name}，请在{self.wait_time_from_config}秒内发送文本描述和可能需要的图片, 然后发送包含'start'或'开始'的消息开始生成。")

    @filter.event_message_type(EventMessageType.ALL)
    async def collect_user_inputs(self, event: AstrMessageEvent):
        """处理后续消息，收集用户输入或触发 /draw 会话的生成。(旧版功能)"""
        if not hasattr(event, 'message_obj') or not hasattr(event.message_obj, 'type'):
             return 

        user_id = event.get_sender_id()
        
        # 忽略机器人自身消息
        if self.robot_id_from_config and user_id == self.robot_id_from_config:
            return

        current_group_id = user_id 
        is_group_message = hasattr(event.message_obj, 'group_id') and event.message_obj.group_id is not None and event.message_obj.group_id != ""
        if is_group_message:
            current_group_id = event.message_obj.group_id
        
        # 白名单检查 (同样应用于后续消息，确保只有授权会话可以继续)
        if self.group_whitelist:
            identifier_to_check = current_group_id if is_group_message else user_id
            if str(identifier_to_check) not in [str(whitelisted_id) for whitelisted_id in self.group_whitelist]:
                return

        current_session_key = (user_id, str(current_group_id))

        # logger.debug(f"collect_user_inputs: Processing message. User ID: {user_id}, Session ID: {current_group_id}, Session Key: {current_session_key}")
        # logger.debug(f"collect_user_inputs: Current waiting users keys: {list(self.waiting_users.keys())}")

        if current_session_key not in self.waiting_users:
            # logger.debug(f"collect_user_inputs: Session key {current_session_key} not found in waiting users. Ignoring message for /draw flow.")
            return # Not a user we are waiting for in the /draw flow

        # logger.debug(f"collect_user_inputs: Session key {current_session_key} IS in waiting users. Processing for /draw flow.")

        if time.time() > self.waiting_users[current_session_key]:
            logger.debug(f"collect_user_inputs: Session {current_group_id} for user {user_id} (key {current_session_key}) timed out for /draw flow.")
            del self.waiting_users[current_session_key]
            if current_session_key in self.user_inputs:
                del self.user_inputs[current_session_key]
            yield event.plain_result("等待超时，您的 /draw 会话已结束。请重新使用 /draw 命令。")
            return

        message_text_raw = event.message_str.strip()
        keywords = ["start", "开始"] # 触发生成的关键词
        # 检查是否包含关键词，同时确保不是在输入另一个命令 (如 /draw 本身)
        contains_keyword = any(keyword in message_text_raw.lower() for keyword in keywords)
        
        # 如果消息是 `/draw` 命令本身，则它应该由 `initiate_creation_session` 处理，而不是在这里收集
        is_command = message_text_raw.startswith("/") or message_text_raw.lower().startswith("draw")
        if is_command and not contains_keyword:
            # logger.debug(f"collect_user_inputs: 消息是命令 ({message_text_raw}) 且不包含 start/开始，已忽略。")
            return


        current_text_for_prompt = message_text_raw
        current_images_pil: List[PILImage.Image] = []

        message_chain = event.get_messages()
        for msg_component in message_chain:
            if isinstance(msg_component, Image) and hasattr(msg_component, 'url') and msg_component.url:
                try:
                    # 旧版使用 download_image_by_url，返回本地路径
                    # 然后用 PILImage.open 打开。
                    # 新版有 download_pil_image_from_url 直接返回 PIL Image。
                    # 为了保持"整块添加"，我们暂时用旧的方式，或者适配到新的。
                    # 适配到新的：
                    pil_img = await self.download_pil_image_from_url(msg_component.url, "用户为/draw会话发送的图片")
                    if pil_img:
                        current_images_pil.append(pil_img)
                        logger.info(f"collect_user_inputs: Successfully downloaded and converted image via new method: {msg_component.url} for /draw session key {current_session_key}")
                    else:
                        yield event.plain_result(f"无法处理您发送的一张图片（下载或转换失败），请尝试其他图片。") # Inform user
                        # return # Optional: stop processing if one image fails
                except Exception as e:
                    logger.error(f"collect_user_inputs: 处理 /draw 会话的图片失败 (key {current_session_key}): {str(e)}", exc_info=True)
                    yield event.plain_result(f"处理图片时发生错误: {str(e)}。请稍后再试或尝试其他图片。")
                    return # Stop processing on error

        # 确保 user_inputs 中有此会话 (理论上 initiate 时已创建)
        if current_session_key not in self.user_inputs:
             logger.error(f"collect_user_inputs: 用户 {user_id} 在会话 {current_group_id} (key {current_session_key}) 中等待，但 user_inputs 状态丢失。正在清理。")
             if current_session_key in self.waiting_users:
                 del self.waiting_users[current_session_key]
             yield event.plain_result("您的 /draw 会话状态异常，请重试。")
             return

        # 存储本次消息的内容
        # 只有在文本或图片非空时才记录，避免空消息污染
        if current_text_for_prompt or current_images_pil:
            message_data = {
              'text': current_text_for_prompt, # Store raw text, keyword removal happens at generation
              'images': current_images_pil,   # Store PIL images
              'timestamp': time.time()
            }
            self.user_inputs[current_session_key]['messages'].append(message_data)
            logger.debug(f"collect_user_inputs: Stored message for /draw session {current_session_key}. Text: '{current_text_for_prompt[:30]}...', Images: {len(current_images_pil)}")


        if contains_keyword:
            logger.debug(f"Gemini_Draw (Command): Start keyword detected in session {current_group_id} (key {current_session_key}). Processing messages.")
            
            # 从 user_inputs 中聚合所有为此会话收集的消息
            collected_session_messages = self.user_inputs[current_session_key].get('messages', [])
            # 按时间戳排序确保顺序
            collected_session_messages.sort(key=lambda x: x['timestamp'])

            all_text_parts = []
            all_pil_images_for_api: List[PILImage.Image] = []

            for msg_data in collected_session_messages:
                text_part = msg_data.get('text', '')
                # 从文本中移除触发关键词，避免它们进入最终的prompt
                for kw in keywords:
                    # Regex to remove whole word, case insensitive
                    text_part = re.sub(r'\b' + re.escape(kw) + r'\b', '', text_part, flags=re.IGNORECASE).strip()
                if text_part:
                    all_text_parts.append(text_part)
                
                all_pil_images_for_api.extend(msg_data.get('images', [])) # images are already PIL

            final_prompt_text = '\n'.join(all_text_parts).strip()

            # 清理会话状态
            del self.waiting_users[current_session_key]
            del self.user_inputs[current_session_key]

            # 如果没有任何用户提供的参考图，则尝试加载默认参考图
            if not all_pil_images_for_api and self.enable_base_reference_image:
                base_image = self._load_base_reference_image()
                if base_image:
                    all_pil_images_for_api.append(base_image)
                    logger.info("已使用默认参考图。")

            if not final_prompt_text and not all_pil_images_for_api:
                yield event.plain_result("您没有提供任何文本描述或图片内容给 /draw 会话。")
                return

            yield event.plain_result("收到开始指令，正在为您生成图片，请稍候...")
            
            try:
                # 调用核心的 API 生成方法
                logger.debug(f"collect_user_inputs: Calling API generate for /draw session (API类型: {self.api_type}). Prompt: '{final_prompt_text[:50]}...', Images: {len(all_pil_images_for_api)}")
                
                # 根据API类型调用相应的生成方法
                if self.api_type == "OpenRouter":
                    api_result = await self.openrouter_generate(final_prompt_text, all_pil_images_for_api)
                else:
                    # 默认使用 Google Gemini API
                    api_result = await self.gemini_generate(final_prompt_text, all_pil_images_for_api)
                
                if api_result is None or not isinstance(api_result, dict): # Should be caught by gemini_generate raising error
                    logger.error(f"collect_user_inputs: gemini_generate 返回无效结果 for /draw session: {type(api_result)}")
                    yield event.plain_result("处理图片时发生内部错误（生成器未返回有效数据）。")
                    return

                text_response = api_result.get('text', '').strip()
                image_paths = api_result.get('image_paths', []) # List of local file paths

                logger.debug(f"collect_user_inputs (/draw): gemini_generate returned - Text: '{text_response[:50]}...', Images: {len(image_paths)}")

                # 缓存机器人自己生成的图片 (对于 /draw 指令，图片的"owner"是触发指令的用户，但图片本身是机器人发的)
                # 如果希望这些图片能被 LLM 工具通过 reference_bot=True 引用，则需要用 robot_id 缓存
                if image_paths and self.robot_id_from_config:
                    logger.info(f"准备缓存 {len(image_paths)} 张 /draw 生成的图片路径到机器人 {self.robot_id_from_config} 在上下文 {current_group_id} 的历史中...")
                    for i, img_path in enumerate(image_paths):
                        if os.path.exists(img_path):
                            self.store_user_image(
                                str(self.robot_id_from_config), # Image belongs to the bot
                                str(current_group_id),        # In the current chat context
                                img_path,                   # Store the local file path
                                f"draw_cmd_generated_{i+1}_{os.path.basename(img_path)}"
                            )

                if not text_response and not image_paths:
                    logger.warning("collect_user_inputs (/draw): API未返回任何文本或图片内容。")
                    yield event.plain_result("未能从API获取任何文本或图片内容。")
                    return

                # 发送结果给用户 (旧版的发送逻辑)
                if len(image_paths) < 2: # 单图或无图（只有文本）
                    chain_to_send = []
                    if text_response:
                        chain_to_send.append(Plain(text_response))
                    for img_path in image_paths:
                        if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            chain_to_send.append(Image.fromFileSystem(img_path))
                    
                    if chain_to_send:
                        yield event.chain_result(chain_to_send)
                    else: # Should not happen if previous checks pass
                        yield event.plain_result("抱歉，未能生成有效内容。")
                else: # 多张图片，使用 Nodes 合并发送
                    bot_id_for_node_str = event.message_obj.self_id or self.robot_id_from_config or self.config.get("bot_id")
                    bot_id_for_node = int(str(bot_id_for_node_str).strip()) if bot_id_for_node_str and str(bot_id_for_node_str).strip().isdigit() else None
                    
                    if bot_id_for_node is None:
                        logger.error("collect_user_inputs (/draw): 无法确定有效的 bot_id 用于合并转发。降级为逐条发送。")
                        if text_response: yield event.plain_result(text_response)
                        for img_path in image_paths:
                            if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                                yield event.chain_result([Image.fromFileSystem(img_path)])
                        return

                    bot_name_for_node = str(self.config.get("bot_name", "绘图助手")).strip() or "绘图助手"
                    
                    # 构建 Nodes
                    # 旧版逻辑是 text_response 分段对应图片，这里简化：先发总文本，再逐个发图片
                    nodes_message_list: List[Node] = []
                    if text_response:
                         nodes_message_list.append(Node(
                            user_id=bot_id_for_node, 
                            nickname=bot_name_for_node, 
                            content=[Plain(text_response)]
                        ))
                    
                    for img_path in image_paths: 
                        if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            # Optionally add a small text like "图片 {idx+1}"
                            # content_for_node = [Plain(f"图片 {idx+1}/{len(image_paths)}"), Image.fromFileSystem(img_path)]
                            content_for_node = [Image.fromFileSystem(img_path)] # Simpler: just image
                            nodes_message_list.append(Node(
                                user_id=bot_id_for_node,
                                nickname=bot_name_for_node,
                                content=content_for_node
                            ))
                    
                    if nodes_message_list:
                        yield event.chain_result([Nodes(nodes_message_list)])
                    else:
                        yield event.plain_result("抱歉，未能生成有效内容进行合并转发。")
                return

            except Exception as e_gen:
                logger.error(f"collect_user_inputs (/draw): 在 /draw 会话的生成或回复阶段发生错误: {str(e_gen)}", exc_info=True)
                yield event.plain_result(f"处理您的 /draw 请求时发生错误: {str(e_gen)}")
                # Ensure session is cleaned up on error too
                if current_session_key in self.waiting_users: del self.waiting_users[current_session_key]
                if current_session_key in self.user_inputs: del self.user_inputs[current_session_key]
                return
        
        else: # 未包含触发关键词，且不是命令
            if current_text_for_prompt.strip() or current_images_pil: 
                logger.debug(f"collect_user_inputs (/draw): 未检测到开始指令 (key {current_session_key})，收到输入: text='{current_text_for_prompt[:30]}...', images_count={len(current_images_pil)}")
                yield event.plain_result("已收到您的输入，请继续发送或发送包含'start'或'开始'的消息结束您的 /draw 会话。")
            # else: (空消息，不回复)
            #    logger.debug(f"collect_user_inputs (/draw): 收到空消息，不含开始指令 (key {current_session_key})，已忽略。")


    async def openrouter_generate(self, text_prompt: str, images_pil: Optional[List[PILImage.Image]] = None):
        """
        调用OpenAI格式的API生成图片。
        支持 OpenRouter、DeepSeek、OpenAI 等兼容 Chat Completions 的服务。
        """
        if not self.api_keys:
            raise ValueError("没有配置API密钥 (api_keys)")
        
        images_pil = images_pil or []
        max_retries = len(self.api_keys)
        last_exception = None
        
        # 确定密钥尝试顺序
        key_indices_to_try = list(range(len(self.api_keys)))
        if self.random_api_key_selection:
            random.shuffle(key_indices_to_try)
        else:
            # 轮询逻辑
            key_indices_to_try = [(self.current_api_key_index + i) % len(self.api_keys) for i in range(len(self.api_keys))]
        
        for attempt_num, key_idx_to_use in enumerate(key_indices_to_try):
            current_key_to_try = self.api_keys[key_idx_to_use]
            
            # --- FIX 1: 在 try 块的一开始就初始化 result，防止 return result 报错 ---
            result = {'text': '', 'image_paths': []}
            
            try:
                logger.info(f"openrouter_generate: 尝试API密钥索引 {key_idx_to_use} (尝试 {attempt_num + 1}/{max_retries})")
                
                # 处理 Base URL
                base_url = self.api_base_url_from_config
                if not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
                    if base_url.endswith('/'):
                        base_url = base_url + 'api/v1'
                    else:
                        base_url = base_url + '/api/v1'
                
                # 初始化客户端
                client = OpenAI(
                    api_key=current_key_to_try,
                    base_url=base_url
                )
                
                # 构建消息内容 (Multimodal / Text)
                message_content = []
                
                # 1. 添加文本提示
                message_content.append({
                    "type": "text",
                    "text": text_prompt
                })
                
                # 2. 添加参考图 (如果存在)
                if images_pil:
                    logger.info(f"将 {len(images_pil)} 张参考图片加入请求上下文")
                    for idx, img in enumerate(images_pil):
                        try:
                            buffered = BytesIO()
                            # 统一转为 PNG 防止格式兼容问题
                            img.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            
                            message_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            })
                        except Exception as e:
                            logger.error(f"处理参考图片 {idx + 1} 失败: {e}")

                logger.info(f"调用 Chat Completions，模型: {self.model_name_from_config}")

                # --- FIX 2: 移除 if is_openrouter 限制，通用化处理 ---
                # 重试机制 (针对生成过程)
                max_generation_retries = 3
                
                for generation_attempt in range(max_generation_retries):
                    try:
                        response = await asyncio.to_thread(
                            client.chat.completions.create,
                            model=self.model_name_from_config,
                            messages=[
                                {
                                    "role": "user",
                                    "content": message_content if len(message_content) > 1 else text_prompt
                                }
                            ]
                        )
                    except Exception as e_gen:
                        logger.warning(f"生成请求失败 (尝试 {generation_attempt+1}): {e_gen}")
                        if generation_attempt == max_generation_retries - 1:
                            raise e_gen
                        await asyncio.sleep(2)
                        continue

                    # 解析响应
                    if response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        message = choice.message
                        
                        raw_content = ""
                        if hasattr(message, 'content') and message.content:
                            raw_content = message.content
                            # 暂存文本，稍后可能还要清洗掉里面的 URL
                            result['text'] = raw_content

                        # --- 图片提取逻辑 ---
                        found_image_urls = []

                        # 方式 A: OpenRouter 专有字段 message.images
                        if hasattr(message, 'images') and message.images:
                            logger.info("检测到 OpenRouter 'images' 字段")
                            for img_item in message.images:
                                if isinstance(img_item, dict) and 'url' in img_item:
                                    found_image_urls.append(img_item['url'])
                                elif isinstance(img_item, str): # 某些情况直接是 URL 列表
                                    found_image_urls.append(img_item)

                        # 方式 B: 从文本内容的 Markdown 或 URL 中提取 (适用于 Nano Banana, Flux 等)
                        # 很多模型直接返回 "Here is your image: ![img](https://...)"
                        if raw_content:
                            # 提取 Markdown 图片: ![...](url)
                            markdown_urls = re.findall(r'!\[.*?\]\((https?://.*?)\)', raw_content)
                            found_image_urls.extend(markdown_urls)
                            
                            # 提取纯 URL (以 http 开头，图片格式结尾) - 作为一个兜底
                            # 简单的正则，避免匹配到括号内的
                            plain_urls = re.findall(r'(https?://[^\s\)]+(?:\.png|\.jpg|\.jpeg|\.webp))', raw_content)
                            for p_url in plain_urls:
                                if p_url not in found_image_urls:
                                    found_image_urls.append(p_url)

                        # --- 下载并保存图片 ---
                        if found_image_urls:
                            logger.info(f"提取到 {len(found_image_urls)} 个图片URL，准备下载...")
                            os.makedirs(self.temp_dir, exist_ok=True)
                            
                            for url in found_image_urls:
                                try:
                                    # 使用 download_pil_image_from_url 复用下载逻辑
                                    # 注意：这里我们是为了拿到本地路径，所以有点绕，
                                    # 但为了稳定性，先下成 PIL 再存是比较稳妥的验证方式，
                                    # 或者直接用 download_file
                                    
                                    # 方法：直接下载为文件
                                    ext = "png"
                                    if ".jpg" in url or ".jpeg" in url: ext = "jpg"
                                    elif ".webp" in url: ext = "webp"
                                    
                                    filename = f"api_gen_{time.time()}_{random.randint(100,999)}.{ext}"
                                    save_path = os.path.join(self.temp_dir, filename)
                                    
                                    await download_file(url=url, path=save_path)
                                    
                                    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                                        result['image_paths'].append(save_path)
                                        logger.info(f"图片下载成功: {save_path}")
                                    else:
                                        logger.warning(f"下载文件为空或失败: {url}")
                                        
                                except Exception as e_dl:
                                    logger.error(f"下载生成的图片失败 {url}: {e_dl}")
                            
                            # 如果成功获取了图片，且文本中全是 URL，可以考虑清空文本避免重复刷屏
                            # 这里简单处理：保留文本，因为 LLM 可能有其他描述
                            break # 跳出生成重试循环
                    
                    logger.warning("本次 API 调用未返回有效图片，准备重试...")
                
                # 如果成功拿到数据（文本或图片），就认为是成功了
                if result['image_paths'] or result['text']:
                    if not self.random_api_key_selection:
                        self.current_api_key_index = (key_idx_to_use + 1) % len(self.api_keys)
                    return result
                    
            except Exception as e:
                logger.error(f"openrouter_generate: API处理失败 (密钥索引 {key_idx_to_use}): {str(e)}", exc_info=True)
                last_exception = e
                
            if attempt_num < max_retries - 1:
                logger.info(f"切换下一个API密钥...")
            else:
                logger.error("所有API密钥尝试完毕，均失败。")
        
        if last_exception:
            raise last_exception
        
        # 如果代码走到这里 result 也是存在的（即便是空的）
        return result

    async def gemini_generate(self, text_prompt: str, images_pil: Optional[List[PILImage.Image]] = None):
        """
        调用Gemini API生成文本和图片。
        支持多API密钥轮询和随机选择。
        """
        if not self.api_keys:
            raise ValueError("没有配置API密钥 (api_keys)")
        images_pil = images_pil or []
        http_options = HttpOptions(base_url=self.api_base_url_from_config)
        max_retries, last_exception = len(self.api_keys), None
        key_indices_to_try = list(range(len(self.api_keys)))
        if self.random_api_key_selection:
            random.shuffle(key_indices_to_try)
        else:
            key_indices_to_try = [(self.current_api_key_index + i) % len(self.api_keys) for i in range(len(self.api_keys))]

        for attempt_num, key_idx_to_use in enumerate(key_indices_to_try):
            current_key_to_try = self.api_keys[key_idx_to_use]
            try:
                logger.info(f"gemini_generate: 尝试API密钥索引 {key_idx_to_use} (尝试 {attempt_num + 1}/{max_retries})")
                client = genai.Client(api_key=current_key_to_try, http_options=http_options)
                contents = []
                if text_prompt:
                    contents.append(text_prompt)
                    # +"。请使用中文回复,文字段与图片对应,除非特意要求，图片中不要有文字。"
                for img_item in images_pil:
                    contents.append(img_item)
                if not contents:
                    raise ValueError("没有有效的内容发送给Gemini API")

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="models/" + self.model_name_from_config,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(response_modalities=['Text', 'Image'])
                )
                result = {'text': '', 'image_paths': []}
                if not response:
                    logger.warning("gemini_generate: API响应为空。")
                    raise ValueError("Gemini API返回空响应。")


                if not hasattr(response, 'candidates') or not response.candidates:
                    logger.warning("gemini_generate: API响应中无候选。")
                    raise ValueError("Gemini API响应中无有效候选。")

                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name == 'SAFETY':
                    s_info = f" 安全评级: {candidate.safety_ratings}" if hasattr(candidate, 'safety_ratings') else ""
                    msg = f"内容因安全策略被阻止 (finish_reason: SAFETY).{s_info}"
                    logger.warning(f"gemini_generate: {msg}")
                    raise genai.types.SafetyFeedbackError(msg)

                if not (hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts):
                    f_info = f"(finish_reason: {candidate.finish_reason.name})" if hasattr(candidate, 'finish_reason') else ""
                    logger.warning(f"gemini_generate: Candidate content/parts为空 {f_info}.")
                    raise ValueError(f"Gemini API返回候选内容或部分为空 {f_info}.")

                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text is not None:
                        result['text'] += part.text
                    elif hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'mime_type') and part.inline_data.mime_type.startswith('image/'):
                        img_data = part.inline_data.data
                        gen_img = PILImage.open(BytesIO(img_data))
                        ext = part.inline_data.mime_type.split('/')[-1]
                        if ext not in ['png', 'jpeg', 'jpg', 'webp', 'gif']:
                            ext = 'png'
                        os.makedirs(self.temp_dir, exist_ok=True)
                        temp_fp = os.path.join(self.temp_dir, f"gemini_gen_{time.time()}_{random.randint(100,999)}.{ext}")
                        gen_img.save(temp_fp)
                        result['image_paths'].append(temp_fp)
                        logger.info(f"Gemini API 生成并保存图片: {temp_fp} (MIME: {part.inline_data.mime_type})")

                if not result['text'] and not result['image_paths']:
                    logger.warning(f"Gemini API返回空文本和图片. Candidate: {candidate}")
                if not self.random_api_key_selection:
                    self.current_api_key_index = (key_idx_to_use + 1) % len(self.api_keys)
                return result
            except Exception as e:
                logger.error(f"gemini_generate: API处理失败 (密钥 {key_idx_to_use}): {str(e)}", exc_info=True)
                last_exception = e

            if attempt_num < max_retries - 1:
                logger.info(f"gemini_generate: 尝试下个API密钥 (下个索引: {key_indices_to_try[attempt_num+1]})")
            else:
                logger.error("gemini_generate: 所有API密钥均尝试失败。")
        if last_exception:
            raise last_exception
        logger.error("gemini_generate: 未能从API获取数据且无明确异常。")
        raise ValueError("Gemini API处理失败，无可用密钥或未记录错误。")

    async def terminate(self):
        """
        插件终止时执行清理操作，包括清空图片缓存和取消后台清理任务。
        """
        logger.info("GeminiArtist: 执行 terminate 清理...")
        # 清理会话状态
        if hasattr(self, 'waiting_users'):
            self.waiting_users.clear()
        if hasattr(self, 'user_inputs'):
            self.user_inputs.clear()
        if hasattr(self, 'image_history_cache'):
            self.image_history_cache.clear()
            logger.info("用户图片URL缓存已清空。")
        if self._background_cleanup_task and not self._background_cleanup_task.done():
            logger.info("取消后台定时清理任务...")
            self._background_cleanup_task.cancel()
            try:
                await self._background_cleanup_task
            except asyncio.CancelledError:
                logger.info("后台清理任务已取消。")
            except Exception as e:
                logger.error(f"等待后台清理任务结束时异常: {e}", exc_info=True)
        else:
            logger.info("无活动后台清理任务或已完成。")
        logger.info(f"最终临时文件清理 ({self.temp_dir})...")
        try:
            await asyncio.to_thread(self._blocking_cleanup_temp_dir_logic, 0)
        except Exception as e:
            logger.error(f"最终清理失败: {e}", exc_info=True)
        # 仅当临时目录是插件特有的且为空时才尝试移除
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir) and self.temp_dir == self.plugin_temp_base_dir:
            try:
                if not os.listdir(self.temp_dir):
                    os.rmdir(self.temp_dir)
                    logger.info(f"已移除空临时目录: {self.temp_dir}")
                else:
                    logger.info(f"临时目录 {self.temp_dir} 非空，未移除。")
            except OSError as e:
                logger.warning(f"移除临时目录 {self.temp_dir} 失败: {e}")
        else:
            logger.info("插件临时目录未找到/定义/非预期，无需移除。")
        logger.info("GeminiArtist: terminate 清理完毕。")
