from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
from PIL import Image
import io
import torch
import gc
import tempfile

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None
model_loaded = False

MODEL_PATHS = {
    'fusion': './visualglm',
    'qwen-vl': 'qwen-vl-chat',
    'hunyuan-vision': 'hunyuan-vision'
}

TYPE_PROMPTS = {
    'general': '描述这张图片',
    'background': '描述这张图片的背景环境',
    'detailed': '详细描述这张图片的所有内容',
    'english': 'Describe this image in English'
}

SYSTEM_PREFIX = "你是LICM，一个轻量化图像描述模型。\n"


def load_model():
    """加载 LICM 模型"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return True
    
    try:
        print("[1/2] 加载 Tokenizer...")
        from transformers import AutoTokenizer
        
        # ===== 本地路径加载模型（默认方式）=====
        model_path = "./visualglm" 
        if not os.path.exists(model_path):
            model_path = "THUDM/visualglm-6b"  
        
        # ===== API加载方式（如需使用，取消下面注释并注释掉上面本地加载代码）=====
        # import requests
        # API_BASE_URL = "http://your-api-server.com/v1"
        # API_KEY = "your-api-key"
        # model_path = "THUDM/visualglm-6b"  # API模式下使用模型名称
        # print(f"  使用API加载模型: {API_BASE_URL}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Tokenizer 加载完成")
        
        print("[2/2] 加载模型...")
        from transformers import AutoModel
        
        # 4-bit 量化加载
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True
            )

            for name, module in model.named_modules():
                if isinstance(module, bnb.nn.Linear4bit):
                    if not hasattr(module.weight, 'quant_state') or module.weight.quant_state is None:
                        module.weight = bnb.nn.Params4bit(
                            module.weight.data,
                            requires_grad=False,
                            quant_type="nf4"
                        ).cuda() if torch.cuda.is_available() else module.weight
            
            print("✓ 动态量化加载完成 (4-bit)")
            
        except Exception as e:
            print(f"! bitsandbytes 加载失败 ({e})，回退到传统方式...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            model = model.quantize(4)
            print("✓ 传统量化完成")
        
        model.eval()
        model_loaded = True
        
        if torch.cuda.is_available():
            print(f"GPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def chat_with_image(image_path, query, history=None):
    """使用模型进行图像对话"""
    global model, tokenizer
    
    if history is None:
        history = []
    
    try:
        prompted_query = SYSTEM_PREFIX + query if not history else query
        
        print(f"[推理] 开始处理图片，查询: {query}")

        with torch.no_grad():
            response, new_history = model.chat(
                tokenizer,
                image_path,
                prompted_query,
                history=history
            )
        
        print(f"[推理] 完成，结果长度: {len(response) if response else 0}")

        if isinstance(response, bytes):
            response = response.decode('utf-8', errors='replace')
        elif isinstance(response, str):
            response = response.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        return response, new_history
        
    except Exception as e:
        print(f"推理错误: {e}")
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
            {'id': 'fusion', 'name': 'VGLM融合模型', 'desc': '自训练本地GLM模型'},
            {'id': 'qwen-vl', 'name': '通义千问VL', 'desc': '阿里云视觉语言大模型'},
            {'id': 'hunyuan-vision', 'name': '腾讯混元Vision', 'desc': '腾讯多模态理解大模型'}
        ]
    })


@app.route('/api/model-status', methods=['GET'])
def api_model_status():
    """API: 获取模型加载状态"""
    global model_loaded
    return jsonify({
        'loaded': model_loaded,
        'message': '模型已在本地加载完成' if model_loaded else '模型尚未加载'
    })


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

        model_name = request.form.get('model', 'fusion')
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
            print(f"[API] 使用模型: {model_name}, 类型: {rec_type}, prompt: {query}")

            response, _ = chat_with_image(tmp_path, query, history=[])
            
            print(f"[API] 识别完成，返回结果")
            
            return jsonify({
                'text': response,
                'model': model_name,
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
        model_name = data.get('model', 'fusion')
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
            print(f"[对话API] 收到图片数据，长度: {len(image_data)}")
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
                # 纯文本对话（可以根据实际情况调整）
                response = "抱歉，我需要有图片才能进行对话。"
                new_history = history
            
            return jsonify({
                'response': response,
                'history': new_history,
                'model': model_name
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
        'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("VGLM Pro Web Server")
    print("=" * 60)
    print("\n模型将在首次请求时自动加载")
    print("访问 http://localhost:5000 使用界面\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
