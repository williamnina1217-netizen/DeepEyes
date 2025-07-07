import json
import requests

from io import BytesIO
from PIL import Image
from tqdm import tqdm
import random
import glob
import io
import time

GLOBAL_TIMEOUT = 30

def check_url_content(url):
    try:
        # 发送 GET 请求
        response = requests.get(url, timeout=GLOBAL_TIMEOUT)
        # 检查请求是否成功
        response.raise_for_status()

        # 检查响应内容是否为空
        if response.content:  # 如果内容非空
            # print("Response content is not empty.")
            return True
        else:
            # print("Response content is empty.")
            return False

    except requests.exceptions.RequestException as e:
        # 捕获任何请求异常
        print(f"Request failed: {e}")
        return False


def compress_image(input_image_path, max_size_kb=10*1024):
    """
    压缩图片并确保其大小不超过指定的KB。

    :param input_image_path: 输入图片的路径
    :param max_size_kb: 压缩后图片的最大大小（KB）
    :return: 压缩后的图片的字节数据
    """
    max_size_bytes = max_size_kb * 1024

    image = None
    if isinstance(input_image_path, str) and input_image_path.startswith('http'):
        # 读http图片
        try_times = 5
        while try_times >= 0:
            response = requests.get(input_image_path, timeout=GLOBAL_TIMEOUT)

            if response.status_code == 200:
                image_data = io.BytesIO(response.content)
                image = Image.open(image_data)
                break
            else:
                try_times -= 1
    else:
        image = Image.open(input_image_path)

    if image is None:
        raise ValueError(f'读取图片失败, 图片地址 = {input_image_path}')

    if image.mode == 'RGBA':
        # print('convert RGB')
        image = image.convert('RGB')

    # 初始质量
    quality = 100
    img_bytes_io = io.BytesIO()

    while True:
        # 重置字节流
        img_bytes_io.seek(0)
        img_bytes_io.truncate(0)

        # 以给定质量保存图片
        image.save(img_bytes_io, format='JPEG', quality=quality)

        # 检查大小是否在限制范围内
        # if img_bytes_io.tell() <= max_size_bytes or quality <= 10:
        img_bytes_pos = img_bytes_io.tell()
        if img_bytes_pos <= max_size_bytes:
            break

        # 减少质量
        quality -= 5

    # print(f' [DEBUG quality] {quality=}')
    img_bytes_io.seek(0)
    return img_bytes_io


def mm_image_search(image_file_id, cursor=0, size=5, max_retry=20):
    """图搜接口

    image_file_id: 图片fileId, 需要通过另一个接口存到桶中获取图片的fileId
    cursor: 翻页
    size: 这一页返回的笔记数量

    curl --location --request POST 'http://agi-redbot.devops.beta.xiaohongshu.com/redbot/backend/api/search/imgSearch' \
    --header 'User-Agent: Apifox/1.0.0 (https://apifox.com)' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "fileId": "xproject/mark/1/guohai/file_66fcfdea-e83d-46e0-8226-41a3481fcb88",
        "cursor": 0,
        "size": 1
    }'

    searchStrategy：textSearch/imageSearch  字段表示文搜/图搜
    """
    # url = 'http://agi-redbot.devops.beta.xiaohongshu.com/redbot/backend/api/search/imgSearch'
    url = 'http://agi-redbot.devops.xiaohongshu.com/redbot/backend/api/search/imgSearch'

    json_payload = {
        'fileId': image_file_id,
        'cursor': cursor,
        'size': size,
    }

    payload = json.dumps(json_payload)
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    data = []
    for it in range(max_retry):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=GLOBAL_TIMEOUT)
            result = json.loads(response.text)
            data = result['data']
            if data:
                for d in data:
                    d["type"] = "xhs"
            break
        except Exception as err:
            print(f' [ERROR] mm_image_search failed {it=}: {err=} -- {response.text=}')
            continue
    return data


def mm_web_search(query, filter=3, xhs_size=5, web_size=5, snippet=False, mkt="", max_retry=20):
    """
    https://apifox.com/apidoc/shared-90b3decf-add1-44c8-8a88-d1582fe8692a/api-185854794

    Args:
        query (_type_): _description_
        filter (int, optional): _description_. Defaults to 1.
        xhs_size (int, optional): _description_. Defaults to 5.
        web_size (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """

    url = "https://agi-redbot.devops.xiaohongshu.com/redbot/backend/api/search/hybrid_web_search"
    # url = "https://agi-redbot.devops.beta.xiaohongshu.com/redbot/backend/api/search/hybrid_web_search"
    json_payload = {
        "query": query,
        "xhs": {
            "size": xhs_size,
            "filters": filter,  # 1 图文，2 视频，3 视频+图片
            "threshold": 0.285,
        },
        "web": {
            "size": web_size,
            "snippet": snippet,
            "crawlTimeOut": 5000,
            "contentLength": -1,
            "mkt": "en-US",
        },
        "politic": None,
    }
    if mkt == "en":
        json_payload["web"]["mkt"] = "en-US"
    payload = json.dumps(json_payload)
    headers = {
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
    }

    result = {"data": []}
    for it in range(max_retry):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=GLOBAL_TIMEOUT)
            result = json.loads(response.text)
            break
        except Exception as err:
            print(f' [ERROR] mm_web_search failed {it=}: {err}')
            continue
    return result["data"]


def upload_to_get_fileId(file_path, max_size_mb=10, try_times=20):
    """上传图片获取fileId

    file_path: 图片路径 或 BytesIO对象
    max_size_mb: 压缩后图片的最大大小（MB）
    try_times: 重试次数

    Returns:
        fileId: 成功上传后返回的fileId
    """
    # url = 'http://agi-gra-render.devops.sit.xiaohongshu.com/dodo/file/update'
    url = 'http://agi-redbot.devops.xiaohongshu.com/dodo/file/update'

    data = {
        "file_type": '1',
        "owner_id": "fengyuan",
        "scene": "mark"
    }

    while try_times >= 0:
        try:
            compressed_img_bytes_io = compress_image(file_path, 1024 * max_size_mb)

            files = {
                'file': compressed_img_bytes_io
            }
            response = requests.post(url, files=files, data=data, timeout=GLOBAL_TIMEOUT)
            compressed_img_bytes_io.close()    # 关闭字节流
            result = json.loads(response.text)

            if result['success']:
                fileId = result['data']['fileId']

                # 再次检查是否写入成功
                check_resp = get_url_from_fileId(fileId)
                if check_resp is not None and check_url_content(check_resp):
                    return fileId
                else:
                    try_times -= 1
            else:
                print('upload fails.')
                try_times -= 1
        except Exception as err:
            print(f'upload error : {err}')
            try_times -= 1

        time.sleep(1)
    return None


def get_url_from_fileId(fileId, try_times=4):
    """根据fileId获取临时图片链接
    """
    # url = 'http://agi-gra-render.devops.sit.xiaohongshu.com/dodo/file/sign'
    url = 'http://agi-redbot.devops.xiaohongshu.com/dodo/file/sign'

    # data = 'xproject/mark/1/guohai/file_12dcadea-6fdf-4c12-9d59-922e38c19b7f'
    data = fileId

    headers = {
        'Content-Type': 'application/json'
    }

    while try_times >= 0:
        try:
            response = requests.post(url, headers=headers, data=data, timeout=GLOBAL_TIMEOUT)
            result = json.loads(response.text)

            if result['success']:
                return result['data']
            else:
                try_times -= 1
        except:
            try_times -= 1

    return None


def set_default_functioncall_args(functioncall):
    if functioncall["name"] in ["text_search", "text_search_bing"]:
        if "size" not in functioncall:
            functioncall["arguments"]["size"] = 3
        if "snippet" not in functioncall:
            functioncall["arguments"]["snippet"] = False
        if "filters" not in functioncall:
            functioncall["arguments"]["filters"] = 1
        if "mkt" not in functioncall:
            functioncall["arguments"]["mkt"] = ""
    else:
        raise ValueError(f"Unknown function name: {functioncall['name']}")
    return functioncall


def execute_text_search(functioncall):
    functioncall = set_default_functioncall_args(functioncall)
    observation = []
    if functioncall["name"] == "text_search":
        search_key = functioncall["arguments"]["query"]
        function_results = mm_web_search(
            query=search_key,
            xhs_size=functioncall["arguments"]["size"],
            web_size=0,
            snippet=functioncall["arguments"]["snippet"],
            filter=functioncall["arguments"]["filters"],
            mkt=functioncall["arguments"]["mkt"],
        )
        for result in function_results:
            result["search_key"] = search_key
        observation = function_results

    if functioncall["name"] == "text_search_bing":
        search_key = functioncall["arguments"]["query"]
        function_results = mm_web_search(
            query=search_key,
            xhs_size=0,
            web_size=functioncall["arguments"]["size"],
            snippet=functioncall["arguments"]["snippet"],
            filter=functioncall["arguments"]["filters"],
            mkt=functioncall["arguments"]["mkt"],
        )
        for result in function_results:
            result["search_key"] = search_key
        observation = function_results
        
    return observation


def execute_image_search(query_image, pbar=None, **kwargs):
    """image search API
    Args:
        query_image: PIL.Image or image file path
        kwargs: 其他参数

    Returns:
        observation: 包含搜索结果的列表
    """
    observation = []
    if isinstance(query_image, str):
        query_image = Image.open(query_image)
    
    if not isinstance(query_image, Image.Image):
        raise ValueError("query_image must be a PIL.Image or a valid image file path")

    image_bytes = io.BytesIO()
    query_image.save(image_bytes, format='PNG')
    fileid = upload_to_get_fileId(image_bytes)
    if fileid is None:
        print(f' [ERROR] upload_to_get_fileId failed ...')
        return observation

    size = kwargs.get("size", 5)
    max_retry = kwargs.get("max_retry", 4)
    search_results = []
    for it in range(max_retry):
        search_results = mm_image_search(fileid, size=size)
        if search_results is None or len(search_results) == 0:
            print(f' [ERROR {it=}] mm_image_search returned empty results for fileId={fileid}')
            continue
        break

    observation.extend(search_results)
    if pbar is not None:
        pbar.update(1)
    return observation


def unit_test_image_api():
    image_url = "https://down-tw.img.susercontent.com/file/tw-11134208-7ras9-m52nutancfsxf8"
    response = requests.get(image_url, timeout=GLOBAL_TIMEOUT)
    image = Image.open(BytesIO(response.content))
    obslist = execute_image_search(image, size=10)

    obsjson = json.dumps(obslist)
    print(f' [DEBUG] obs={obsjson}')
    for ix, obs in enumerate(obslist):
        if "id" in obs:
            print(f' [DEBUG {ix}] obs id={obs["id"]}')
        if "title" in obs:
            print(f' [DEBUG {ix}] obs title={obs["title"]}')
        if "content" in obs:
            print(f' [DEBUG {ix}] obs content={obs["content"]}')
        if "url" in obs:
            print(f' [DEBUG {ix}] obs url={obs["url"]}')


def unit_test_image_api2():
    image_pattern = "/cpfs/user/fengyuan/code/github/zero-rl-data/geoguessr/deboradum-geogeussr/train/*/*.png"
    image_fns = glob.glob(image_pattern)
    random.shuffle(image_fns)
    image_fns = image_fns[:1000]

    num_workers = 1
    pbar = tqdm(
        total=len(image_fns),
        desc=f'Image searching on {num_workers} workers'
    )

    obs_list = []
    if num_workers <= 1:
        for image_fn in image_fns:
            obs = execute_image_search(image_fn, pbar=pbar)
            obs_list.append(obs)
            print(json.dumps(obs, ensure_ascii=False, indent=2))
            exit()
    else:
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        partial_func = partial(execute_image_search, pbar=pbar)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            obs_list = list(executor.map(partial_func, image_fns))
    
    pbar.close()
    print('Done!!')


if __name__ == "__main__":
    # obs = mm_web_search("初恋男友忽冷忽热 分分合合的感情还能持续多久", filter=3)
    # for line in obs:
    #     print(line["relevanceScore"])
    #     print(line["url"])

    unit_test_image_api()
    # unit_test_image_api2()
