#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
from PIL import Image
import io
import torch
import gc
import tempfile
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None
model_loaded = False

# 默认本地模型路径
DEFAULT_MODEL_PATH = "/gemini/pretrain/checkpoints/finetune-visualglm-6b-03-04-23-02/11300"

# API 配置（如需使用API模型，取消下面注释并配置）
# API_CONFIG = {
#     'qwen-vl': {
#         'base_url': 'https://dashscope.aliyuncs.com/api/v1',
#         'api_key': 'your-dashscope-api-key',
#         'model_name': 'qwen-vl-chat'
#     },
#     'hunyuan-vision': {
#         'base_url': 'https://hunyuan.tencentcloudapi.com',
#         'api_key': 'your-tencent-api-key',
#         'model_name': 'hunyuan-vision'
#     }
# }

TYPE_PROMPTS = {
    'general': '描述这张图片',
    'background': '描述这张图片的背景环境',
    'detailed': '详细描述这张图片的所有内容',
    'english': 'Describe this image in English'
}

SYSTEM_PREFIX = "你是VGLM，一个图像描述模型。\n"

use_api_mode = False
api_client = None
api_model_name = None

def load_model(model_path=None, use_quant=False, quant_bits=4, use_api=False, api_config=None):
    """加载模型 (SAT框架或API)
    
    Args:
        model_path: 模型路径，默认使用训练好的模型
        use_quant: 是否使用量化
        quant_bits: 量化位数 (4 或 8)
        use_api: 是否使用API模式
        api_config: API配置字典
    """
    global model, tokenizer, model_loaded, use_api_mode, api_client, api_model_name
    
    if model_loaded:
        return True
    
    # ===== API加载方式（如需使用，设置use_api=True并传入api_config）=====
    if use_api and api_config:
        try:
            print(f"\n[API模式] 初始化API客户端...")
            print(f"  API地址: {api_config.get('base_url')}")
            print(f"  模型名称: {api_config.get('model_name')}")
            
            # 这里可以实现具体的API客户端初始化
            # 例如：openai.OpenAI(base_url=..., api_key=...)
            
            use_api_mode = True
            api_model_name = api_config.get('model_name')
            model_loaded = True
            print("✓ API客户端初始化完成")
            return True
        except Exception as e:
            print(f"API客户端初始化失败: {e}")
            return False
    
    # ===== 本地模型加载方式（默认方式）=====
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    try:
        print(f"\n[1/2] 加载 Tokenizer...")
        from transformers import AutoTokenizer
        
        # Tokenizer 从 visualglm 目录加载
        tokenizer_path = "./visualglm"
        if not os.path.exists(tokenizer_path):
            tokenizer_path = "THUDM/visualglm-6b"
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Tokenizer 加载完成")
        
        print(f"\n[2/2] 加载模型 (SAT)...")
        print(f"  模型路径: {model_path}")
        
        if use_quant:
            print(f"  使用 {quant_bits}-bit 量化")
        
        from sat.model import AutoModel
        from sat.model.mixins import CachedAutoregressiveMixin
        from sat.quantization.kernels import quantize
        from finetune_visualglm import FineTuneVisualGLMModel

        model, model_args = AutoModel.from_pretrained(
            model_path,
            args=argparse.Namespace(
                fp16=True,
                skip_init=True,
                use_gpu_initialization=True if (torch.cuda.is_available() and not use_quant) else False,
                device='cuda' if (torch.cuda.is_available() and not use_quant) else 'cpu',
            )
        )
        model = model.eval()
        
        # 量化
        if use_quant and quant_bits in [4, 8]:
            quantize(model, quant_bits)
            if torch.cuda.is_available():
                model = model.cuda()
        
        # 添加自动回归 mixin
        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        
        print("✓ 模型加载完成")
        
        if torch.cuda.is_available():
            print(f"\nGPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        use_api_mode = False
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def chat_with_image(image_path, query, history=None, model_name=None):
    """使用模型进行图像对话 (SAT框架或API)"""
    global model, tokenizer, use_api_mode, api_client, api_model_name
    
    if history is None:
        history = []
    
    # ===== API模式推理 =====
    if use_api_mode:
        try:
            print(f"[API推理] 开始处理图片，查询: {query}")
            
            # 这里实现具体的API调用逻辑
            # 示例：使用OpenAI兼容的API格式
            # import openai
            # with open(image_path, 'rb') as img_file:
            #     response = api_client.chat.completions.create(
            #         model=api_model_name,
            #         messages=[{
            #             "role": "user",
            #             "content": [
            #                 {"type": "text", "text": query},
            #                 {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"}
            #             ]
            #         }]
            #     )
            # result = response.choices[0].message.content

            result = f"[API模式] 请实现具体的API调用逻辑\n查询: {query}\n模型: {api_model_name}"
            
            print(f"[API推理] 完成")
            return result, history
            
        except Exception as e:
            print(f"API推理错误: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    # ===== 本地模型推理 =====
    try:
        prompted_query = SYSTEM_PREFIX + query if not history else query
        
        print(f"[推理] 开始处理图片，查询: {query}")

        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import chat

        with torch.no_grad():
            response, new_history, cache_image = chat(
                image_path,
                model,
                tokenizer,
                prompted_query,
                history=history,
                image=None,
                max_length=256,
                top_p=0.8,
                temperature=0.8,
                top_k=100,
                english=False,
                invalid_slices=[]
            )
        
        print(f"[推理] 完成，结果长度: {len(response) if response else 0}")

        if isinstance(response, bytes):
            response = response.decode('utf-8', errors='replace')
        elif isinstance(response, str):
            response = response.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        return response, new_history
        
    except Exception as e:
        print(f"推理错误: {e}")
        import traceback
        traceback.print_exc()
        raise e


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def api_models():
    """API: 获取可用模型列表"""
    return jsonify({
        'models': [
            {'id': 'finetuned', 'name': 'VGLM微调模型', 'desc': '基于COCO数据集微调的模型'},
            {'id': 'base', 'name': 'VGLM基础模型', 'desc': '原始VisualGLM-6B模型'},
            {'id': 'qwen-vl', 'name': '通义千问VL', 'desc': '阿里云视觉语言大模型 (API)'},
            {'id': 'hunyuan-vision', 'name': '腾讯混元Vision', 'desc': '腾讯多模态理解大模型 (API)'}
        ]
    })


@app.route('/api/model-status', methods=['GET'])
def api_model_status():
    """API: 获取模型加载状态"""
    global model_loaded
    return jsonify({
        'loaded': model_loaded,
        'model_path': DEFAULT_MODEL_PATH if model_loaded else None,
        'message': '模型已在本地加载完成' if model_loaded else '模型尚未加载'
    })


@app.route('/api/load-model', methods=['POST'])
def api_load_model():
    """API: 手动加载模型"""
    global model_loaded, use_api_mode
    
    try:
        data = request.json
        model_id = data.get('model_id', 'finetuned')
        model_path = data.get('model_path', DEFAULT_MODEL_PATH)
        use_quant = data.get('use_quant', False)
        quant_bits = data.get('quant_bits', 4)
        
        if model_loaded:
            return jsonify({'success': True, 'message': '模型已经加载'})
        
        # ===== API模型加载 =====
        if model_id in ['qwen-vl', 'hunyuan-vision']:
            # 取消下面的注释并配置API_CONFIG以启用API模式
            # api_config = API_CONFIG.get(model_id)
            # if api_config:
            #     success = load_model(use_api=True, api_config=api_config)
            # else:
            #     return jsonify({'success': False, 'error': f'未找到 {model_id} 的API配置'}), 400
            
            # 临时返回提示
            return jsonify({
                'success': False, 
                'error': f'API模式需要在代码中配置 {model_id} 的API密钥和地址',
                'hint': f'请取消 app_sat.py 中 API_CONFIG 的注释并配置 {model_id} 的参数'
            }), 400
        
        # ===== 本地模型加载 =====
        success = load_model(model_path, use_quant, quant_bits)
        
        if success:
            return jsonify({
                'success': True,
                'message': '模型加载成功',
                'model_path': model_path,
                'is_api': use_api_mode,
                'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() and not use_api_mode else None
            })
        else:
            return jsonify({'success': False, 'error': '模型加载失败'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API: 识别图片"""
    global model_loaded
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        rec_type = request.form.get('type', 'general')

        if not model_loaded:
            print("首次请求，加载模型...")
            success = load_model()
            if not success:
                return jsonify({'error': '模型加载失败，请检查模型路径'}), 500
        
        # 保存上传的图片到临时文件
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name, 'JPEG')
            tmp_path = tmp_file.name
        
        try:
            query = TYPE_PROMPTS.get(rec_type, '描述这张图片')
            print(f"[API] 类型: {rec_type}, prompt: {query}")

            response, _ = chat_with_image(tmp_path, query, history=[])
            
            print(f"[API] 识别完成")
            
            return jsonify({
                'text': response,
                'type': rec_type
            })
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        print(f"API 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API: 对话问答"""
    global model_loaded
    
    try:
        data = request.json
        message = data.get('message', '')
        history = data.get('history', [])
        image_data = data.get('image')  
        
        if not message:
            return jsonify({'error': '消息不能为空'}), 400

        if not model_loaded:
            print("首次请求，加载模型...")
            success = load_model()
            if not success:
                return jsonify({'error': '模型加载失败，请检查模型路径'}), 500

        image_path = None
        if image_data:
            print(f"[对话API] 收到图片数据")
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            print(f"[对话API] 图片解码成功，尺寸: {image.size}")
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file.name, 'JPEG')
                image_path = tmp_file.name
        
        try:
            if image_path:
                response, new_history = chat_with_image(image_path, message, history=history)
            else:
                response = "抱歉，我需要有图片才能进行对话。"
                new_history = history
            
            return jsonify({
                'response': response,
                'history': new_history
            })
            
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        print(f"对话 API 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """API: 获取模型状态"""
    return jsonify({
        'model_loaded': model_loaded,
        'cuda_available': torch.cuda.is_available(),
        'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else None,
        'default_model_path': DEFAULT_MODEL_PATH
    })


if __name__ == '__main__':
    print("=" * 60)
    print("VGLM Web Server (SAT框架)")
    print("=" * 60)
    print(f"\n默认模型路径: {DEFAULT_MODEL_PATH}")
    print("模型将在首次请求时自动加载")
    print("访问 http://localhost:5000 使用界面\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
