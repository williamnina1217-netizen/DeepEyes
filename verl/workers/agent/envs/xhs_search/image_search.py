import numpy as np
import copy
import requests
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from math import ceil, floor
from io import BytesIO
# 临时修复
# ToolBase.registry = {}

from verl.workers.agent.tool_envs import ToolBase
from .function_call import execute_text_search, execute_image_search

class ImageSearch(ToolBase):
    name = "image_search"

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed
        self.is_multi_modal = False

    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None

    def extract_action(self, action_string: str) -> Dict[str, Any]:
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}

        try:
            tool_call = json.loads(action.strip())  # 或使用 literal_eval
        except Exception as e:
            error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
            obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n<think>"
            info = {"error": str(e), "status": "failed"}
            return obs, 0.0, False, {}

        try:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
        
            if tool_name == "image_zoom_in" and self.is_multi_modal:
                # Zoom in by cropping the image
                # image_path = args["image_path"]
                bbox = args["bbox_2d"]
                bbox = self.maybe_resize_bbox(*bbox)
                if not bbox:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                cropped_img = img.crop(bbox)
                obs_text = "<image>"
                obs_image_list = [cropped_img]
                
            elif tool_name == "image_rotate" and self.is_multi_modal:
                # Rotate the image
                # image_path = args["image_path"]
                angle = args["angle"]
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                rotated_img = img.rotate(angle)
                obs_text = "<image>"
                obs_image_list = [rotated_img]

            elif tool_name in ["text_search", "text_search_bing"]:
                observation = execute_text_search(tool_call)
                obs_text, obs_image_list = self._observation2messages(observation)

            elif tool_name == "image_zoom_in_search" and self.is_multi_modal:
                bbox = args["bbox_2d"]
                bbox = self.maybe_resize_bbox(*bbox)
                if not bbox:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                cropped_img = img.crop(bbox)

                ret_size = args.get("ret_size", 3)
                observation = execute_image_search(cropped_img, size=ret_size)
                obs_text, obs_image_list = self._observation2messages(observation)
                
            else:
                raise ValueError(f"FUNCTION NAME {tool_name} NOT ALLOWED")

            # Prepare the observation
            if len(obs_image_list) > 0:
                obs = {
                    "prompt": "<|im_end|>\n<|im_start|>user\n<tool_response>" + obs_text + "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>",
                    "multi_modal_data": {"image": obs_image_list}
                }
            else:
                obs = "<|im_end|>\n<|im_start|>user\n<tool_response>" + obs_text + "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>"

            reward = 0.0  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG] SUCCESS ACTION {len(obs_image_list)=}, {obs_text=}')
            return obs, reward, done, info

        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[ERROR] Execute WRONG - {str(e)} {action_string=}')
            obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(e)}</tool_response>" + "<|im_end|>\n<|im_start|>assistant\n<think>"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data

        if "image" in self.multi_modal_data and len(self.multi_modal_data["image"]) > 0:
            self.is_multi_modal = True
            self.height = self.multi_modal_data['image'][0].height
            self.width = self.multi_modal_data['image'][0].width
        else:
            self.is_multi_modal = False
            self.height, self.width = 0, 0

    def _observation2messages(self, observation):
        obs_image_list = []
        all_obs_string = ""
        for idx, obs in enumerate(observation):
            this_obs_string = f"""
id: {idx}
title: {obs['title']}
content:\n{obs['content']}
likes: {obs['likes']}
publish_time: {obs['publish_time']}
"""
            cover_url = obs.get("coverUrl", "")
            cover_image = self.maybe_download_image(cover_url)
            if cover_image is not None:
                this_obs_string = "<image>" + this_obs_string
                obs_image_list.append(cover_image)
            all_obs_string += this_obs_string

        print(f' [DEBUG search obs] {all_obs_string}')
        return all_obs_string, obs_image_list

    def maybe_download_image(self, image_url):
        if not image_url:
            return None
        if not isinstance(image_url, str):
            return None

        # if image_url in ToolBase.shared_mapping.keys():
        #     return ToolBase.shared_mapping[image_url]

        try:
            response = requests.get(image_url, timeout=30)
            image = Image.open(BytesIO(response.content))
        except Exception as err:
            print(f' [ERROR] download image error : {err}')
            return None
        # ToolBase.shared_mapping[image_url] = image
        return image

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            assert min(height, width) > 30, f"{height=}, {width=} is too small"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False

    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualToolBox("visual_toolbox", "Tool for image processing", {})
    
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"image_path": "test.jpg", "bbox": [10, 10, 100, 100]}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    
    # Test rotate tool (should return reward=0.1)
    rotate_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(rotate_action)
    print(f"Rotate result - Reward: {reward}, Info: {info}")
    
    # Test invalid JSON (should return reward=0.0)
    invalid_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(invalid_action)
    print(f"Invalid JSON result - Reward: {reward}, Info: {info}")
    
    # Test unknown tool (should return reward=0.0)
    unknown_tool_action = """
    <tool_call>
    {"name": "unknown_tool", "arguments": {"param": "value"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(unknown_tool_action)
    print(f"Unknown tool result - Reward: {reward}, Info: {info}")
