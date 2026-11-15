import base64
import json
import logging
import os
import random
import struct
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from cv2.typing import MatLike
import numpy as np
import onnxruntime as ort
import cv2

logger = logging.getLogger(__name__)


def find_model_file(filename: str) -> str:
    """查找模型文件，支持多个路径自动探测。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "..", "models", filename),
        os.path.join(base_dir, "..", filename),
        os.path.join(os.getcwd(), "models", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"找到模型文件: {path}")
            return path
    default_path = os.path.join(base_dir, "..", "models", filename)
    logger.warning(f"未找到模型文件 {filename}，使用默认路径: {default_path}")
    return default_path


def calculate_precise_slider_distance(target_start_x: int, target_end_x: int,
                                      slider_width: int) -> float:
    try:
        target_center_x = (target_start_x + target_end_x) / 2
        slider_initial_center_x = slider_width / 2
        precise_distance = target_center_x - slider_initial_center_x
        random_offset = random.uniform(-0.1, 0.1)
        final_distance = round(precise_distance + random_offset, 1)
        logger.info(f"计算滑块距离成功: {final_distance}")
        return final_distance
    except Exception as e:
        logger.error(f"计算滑块距离时发生错误: {e}")
        raise


def extract_png_width(png_binary: bytes) -> int:
    try:
        if png_binary[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("无效的PNG签名：不是有效的PNG图像")
        width = struct.unpack(">I", png_binary[16:20])[0]
        logger.info(f"提取PNG宽度成功: {width}")
        return width
    except struct.error as e:
        logger.error(f"无法从PNG数据中提取宽度信息: {e}")
        raise ValueError("无法从PNG数据中提取宽度信息") from e
    except Exception as e:
        logger.error(f"提取PNG宽度时发生错误: {e}")
        raise


def slide_match(target_bytes: bytes, background_bytes: bytes) -> list:
    try:
        target = cv2.imdecode(np.frombuffer(target_bytes, np.uint8),
                              cv2.IMREAD_ANYCOLOR)
        background = cv2.imdecode(np.frombuffer(background_bytes, np.uint8),
                                  cv2.IMREAD_ANYCOLOR)
        background = cv2.Canny(background, 100, 200)
        target = cv2.Canny(target, 100, 200)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
        res = cv2.matchTemplate(background, target, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        h, w = target.shape[:2]
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        logger.info(f"滑块匹配成功，最大相似度: {max_val}")
        return [int(max_loc[0]), int(bottom_right[0])]
    except Exception as e:
        logger.error(f"滑块匹配时发生错误: {e}")
        raise


def recognize_blockPuzzle_captcha(target: str, background: str) -> str:
    try:
        target_bytes = base64.b64decode(target)
        background_bytes = base64.b64decode(background)
        res = slide_match(target_bytes=target_bytes,
                          background_bytes=background_bytes)
        target_width = extract_png_width(target_bytes)
        slider_distance = calculate_precise_slider_distance(
            res[0], res[1], target_width)
        slider_data = {
            "x": slider_distance,
            "y": 5,
        }
        logger.info(f"验证码识别成功: {slider_data}")
        return json.dumps(slider_data, separators=(",", ":"))
    except Exception as e:
        logger.error(f"验证码识别时发生错误: {e}")
        raise


_tesseract_available = None
_tesseract_configured = False


def _configure_tesseract():
    global _tesseract_configured
    if _tesseract_configured:
        return
    try:
        import pytesseract
        import shutil
        tpath = shutil.which('tesseract')
        if tpath:
            pytesseract.pytesseract.tesseract_cmd = tpath
        _tesseract_configured = True
    except Exception:
        pass


def is_tesseract_available() -> bool:
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    try:
        _configure_tesseract()
        import pytesseract
        pytesseract.get_tesseract_version()
        _tesseract_available = True
    except Exception:
        _tesseract_available = False
    return _tesseract_available


def predict_ocr_tesseract(image: np.ndarray) -> str:
    _configure_tesseract()
    import pytesseract
    from PIL import Image as PILImage
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb_img)
    text = pytesseract.image_to_string(
        pil_img, lang='chi_sim',
        config='--psm 10 --oem 3'
    ).strip()
    text = text.replace(' ', '').replace('\n', '').replace('\r', '')
    return text


def detect_objects(model_path: str,
                   image_data: MatLike,
                   use_gpu: bool = False) -> list[list[int]]:
    try:
        scale = min(640 / image_data.shape[1], 640 / image_data.shape[0])
        img_resized = cv2.resize(
            image_data,
            (int(image_data.shape[1] * scale), int(
                image_data.shape[0] * scale)),
        )
        new_image = np.full((640, 640, 3), 128, dtype=np.uint8)
        dh, dw = (640 - img_resized.shape[0]) // 2, (640 -
                                                     img_resized.shape[1]) // 2
        new_image[dh:dh + img_resized.shape[0],
                  dw:dw + img_resized.shape[1]] = (img_resized)
        input_img = (np.expand_dims(new_image.transpose(
            (2, 0, 1)), axis=0).astype(np.float32) / 255.0)
        providers = ["CUDAExecutionProvider"
                     ] if use_gpu else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        result = session.run(None, {session.get_inputs()[0].name: input_img})
        detections = result[0][0]
        boxes = [[
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ] for x_center, y_center, width, height, confidence, *class_scores in
                 detections if confidence >= 0.5]
        scores = [
            max(class_scores)
            for _, _, _, _, confidence, *class_scores in detections
            if confidence >= 0.5
        ]
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
            boxes = [boxes[i] for i in indices]
        img_h, img_w = image_data.shape[:2]
        valid_boxes = []
        for x1, y1, x2, y2 in boxes:
            nx1 = max(0, int((x1 - dw) / scale))
            ny1 = max(0, int((y1 - dh) / scale))
            nx2 = min(img_w, int((x2 - dw) / scale))
            ny2 = min(img_h, int((y2 - dh) / scale))
            if nx2 > nx1 and ny2 > ny1:
                valid_boxes.append([nx1, ny1, nx2, ny2])
        return valid_boxes
    except Exception as e:
        raise ValueError(f"目标检测失败: {e}")


def predict_ocr(model_path: str,
                image: np.ndarray,
                use_gpu: bool = False) -> str:
    try:
        session = ort.InferenceSession(
            model_path,
            providers=(["CUDAExecutionProvider"]
                       if use_gpu else ["CPUExecutionProvider"]),
        )
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("输入图片为空或尺寸无效")
        image = np.expand_dims(
            cv2.cvtColor(cv2.resize(image,
                                    (64, 64)), cv2.COLOR_BGR2RGB).transpose(
                                        (2, 0, 1)).astype(np.float32) / 255.0,
            axis=0,
        )
        charset = [
            "士","候","之","科","孩","雪","万","章","导","治",
            "亲","社","所","似","验","习","吃","历","写","业",
            "为","睛","睡","将","林","法","你","观","信","掉",
            "觉","站","确","老","方","道","海","性","好","感",
            "女","术","如","重","细","青","流","心","包","越",
            "且","风","哥","菜","劳","必","阶","代","令","志",
            "国","们","记","知","谁","讲","眼","提","由","民",
            "怎","度","村","没","呀","许","以","四","政","点",
            "离","说","带","关","答","出","放","告","夜","识",
            "兴","做","难","八","叶","月","马","办","行","三",
            "最","小","亮","作","晚","义","活","公","旁","色",
            "看","从","话","系","高","水","您","到","装","中",
            "研","雨","住","因","少","原","什","片","准","脚",
            "张","深","力","让","顶","石","山","类","野","阵",
            "赶","见","七","立","整","屋","再","读","相","弟",
            "两","接","种","车","近","外","几","停","认","特",
            "战","化","子","定","边","多","产","形","她","衣",
            "共","音","分","级","别","千","连","理","往","先",
            "队","围","满","在","领","画","他","反","花","农",
            "被","名","这","席","众","很","渐","乡","极","实",
            "城","取","题","儿","响","那","主","进","去","思",
            "找","总","应","船","身","牛","歌","团","爬","岁",
            "着","冲","早","利","受","忽","苦","也","表","通",
            "有","像","现","对","头","开","般","呼","又","的",
            "把","帮","收","军","怕","饭","或","就","年","背",
            "来","革","压","斗","位","房","飞","都","块","跳",
            "变","今","命","区","爱","门","入","九","动","根",
            "南","造","其","者","便","每","事","座","算","然",
            "笑","阳","半","大","是","会","一","非","树","旧",
            "里","至","无","问","发","河","物","东","叔","它",
            "百","拿","叫","明","刚","脸","干","样","呢","更",
            "底","忙","我","结","地","界","草","论","还","轻",
            "数","世","只","用","长","个","光","此","沙","面",
            "白","转","哪","想","件","文","未","啦","口","十",
            "人","各","并","敌","打","古","合","完","啊","线",
            "回","嘴","究","岸","听","内","土","跑","日","平",
            "咱","快","坚","真","够","工","些","已","争","得",
            "望","伟","却","处","但","过","唱","时","热","走",
            "书","不","起","神","使","本","自","倒","比","前",
            "新","直","经","解","步","胜","次","该","六","后",
            "报","体","家","急","际","五","北","等","员","何",
            "火","吗","机","当","么","天","枪","量","意","同",
            "决","钱","情","手","强","全","了","可","果","气",
            "加","学","息","黑","刻","而","慢","紧","照","指",
            "改","上","运","声","二","吧","己","字","才","教",
            "于","向","要","建","展","句","史","给","坐","和",
            "第","成","落","跟","群","星","生","部","送","服",
            "穿","友","下","拉","任","太","常","场","敢","清",
            "路","破","传","空","师","切","条",
        ]
        return "".join(charset[item] for item in session.run(
            None, {session.get_inputs()[0].name: image})[1])
    except Exception as e:
        raise Exception(f"OCR预测失败: {e}")


def save_click_word_snapshot(image: np.ndarray, bboxes: list, click_points: list,
                             wordlist: list, recognized_dict: dict,
                             stage: str = "predict") -> str:
    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = None
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(fp):
            font_path = fp
            break
    try:
        font_label = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
        font_click = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
    except Exception:
        font_label = ImageFont.load_default()
        font_click = ImageFont.load_default()
    for word, bbox in recognized_dict.items():
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, y1 - 18), word, fill=(0, 255, 0), font=font_label)
    for i, pt in enumerate(click_points):
        x, y = pt["x"], pt["y"]
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
        label = f"{i + 1}.{wordlist[i] if i < len(wordlist) else '?'}"
        draw.text((x + 10, y - 14), label, fill=(255, 0, 0), font=font_click)
    annotated = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(snapshot_dir,
                            f"click_word_{stage}_{timestamp}.png")
    cv2.imwrite(filename, annotated)
    logger.info(f"文字点选快照已保存: {filename}")
    return filename


def recognize_clickWord_captcha(target: str, wordlist: list) -> str:
    target_bytes = base64.b64decode(target)
    image = cv2.imdecode(np.frombuffer(target_bytes, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    bboxes = detect_objects(find_model_file("yolov5n.onnx"), image)
    ocr_model_path = find_model_file("ocr.onnx")
    use_onnx_ocr = os.path.exists(ocr_model_path) and os.path.getsize(ocr_model_path) > 13000000
    if not use_onnx_ocr:
        logger.info("ocr.onnx 模型不可用，尝试使用Tesseract OCR")
        if not is_tesseract_available():
            raise Exception("OCR不可用：既没有ocr.onnx模型，Tesseract也不可用")
    recognized_dict = {}
    for bbox in bboxes:
        try:
            x_min, y_min, x_max, y_max = bbox
            crop_img = image[y_min:y_max, x_min:x_max]
            if use_onnx_ocr:
                text = predict_ocr(ocr_model_path, crop_img)
            else:
                text = predict_ocr_tesseract(crop_img)
            if text:
                recognized_dict[text] = bbox
                logger.info(f"OCR识别: [{text}] @ ({x_min},{y_min})-({x_max},{y_max})")
        except Exception as e:
            logger.warning(f"处理文本框时出错: {e}")
    logger.info(f"需要点击: {wordlist}")
    logger.info(f"识别结果: {list(recognized_dict.keys())}")
    random_coordinates = []
    for word in wordlist:
        bbox = recognized_dict.get(word)
        if bbox:
            x = random.randint(bbox[0], bbox[2])
            y = random.randint(bbox[1], bbox[3])
            random_coordinates.append({"x": x, "y": y})
        else:
            logger.warning(f"未找到字符: {word}")
    save_click_word_snapshot(image, bboxes, random_coordinates, wordlist,
                             recognized_dict, stage="predict")
    return json.dumps(random_coordinates, separators=(",", ":"))
