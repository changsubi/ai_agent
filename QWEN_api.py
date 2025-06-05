from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from datetime import datetime

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

class QwenClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")
        
    def load_model(self):
        print("Qwen3 모델 로딩 중...")
        
        try:
            # Qwen 모델 로드
            model_name = "Qwen/Qwen2.5-3B-Instruct"  # 또는 더 작은 모델
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                load_in_8bit=True if self.device == "cuda" else False
            )
            
            print("✅ Qwen3 모델 로딩 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {str(e)}")
            print("대체 모델로 시도...")
            
            try:
                # 더 작은 모델로 대체
                model_name = "Qwen/Qwen2.5-1.5B-Instruct"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                print("✅ 대체 모델 로딩 완료!")
                return True
                
            except Exception as e2:
                print(f"❌ 대체 모델도 실패: {str(e2)}")
                return False
    
    def classify_schedule_intent(self, text):
        """텍스트에서 일정 관련 의도를 분류"""
        
        system_prompt = """당신은 일정 관리 시스템의 의도 분류 전문가입니다.
사용자의 음성 텍스트를 분석하여 다음 4가지 작업 중 하나를 선택하세요:

<일정 생성> - 새로운 일정이나 약속을 만들어 달라는 요청
<일정 등록> - 기존에 정해진 일정을 시스템에 추가하라는 요청  
<일정 수정> - 기존 일정의 시간, 내용, 장소 등을 바꾸라는 요청
<일정 삭제> - 기존 일정을 취소하거나 제거하라는 요청

반드시 위 4개 태그 중 하나만 정확히 출력하세요."""

        user_prompt = f"""다음 텍스트를 분석하여 적절한 태그를 선택하세요:

텍스트: "{text}"

분석 결과:"""

        try:
            # 대화 형식으로 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 토크나이즈
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            print(f"원본 응답: {response}")
            
            # 태그 추출
            tag = self.extract_tag(response)
            return tag
            
        except Exception as e:
            print(f"분류 중 오류: {str(e)}")
            return self.fallback_classification(text)
    
    def extract_tag(self, response):
        """응답에서 태그 추출"""
        tags = ["<일정 생성>", "<일정 등록>", "<일정 수정>", "<일정 삭제>"]
        
        # 정확한 태그 매칭
        for tag in tags:
            if tag in response:
                return tag
        
        # 키워드 기반 매칭
        response_lower = response.lower()
        
        if any(keyword in response_lower for keyword in ["생성", "만들", "새로", "추가"]):
            return "<일정 생성>"
        elif any(keyword in response_lower for keyword in ["등록", "입력", "저장"]):
            return "<일정 등록>"
        elif any(keyword in response_lower for keyword in ["수정", "변경", "바꾸", "편집"]):
            return "<일정 수정>"
        elif any(keyword in response_lower for keyword in ["삭제", "취소", "제거", "지우"]):
            return "<일정 삭제>"
        
        return "<일정 생성>"  # 기본값
    
    def fallback_classification(self, text):
        """간단한 규칙 기반 분류 (모델 실패시)"""
        text_lower = text.lower()
        
        # 키워드 기반 분류
        if any(keyword in text_lower for keyword in ["만들어", "생성", "새로", "추가해"]):
            return "<일정 생성>"
        elif any(keyword in text_lower for keyword in ["등록", "입력", "저장"]):
            return "<일정 등록>"
        elif any(keyword in text_lower for keyword in ["수정", "변경", "바꿔", "편집"]):
            return "<일정 수정>"
        elif any(keyword in text_lower for keyword in ["삭제", "취소", "제거", "지워"]):
            return "<일정 삭제>"
        else:
            return "<일정 생성>"  # 기본값

# 전역 분류기 인스턴스
classifier = QwenClassifier()

@app.route('/classify', methods=['POST'])
def classify_intent():
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': '텍스트가 필요합니다'
            }), 400
        
        print(f"분류할 텍스트: {text}")
        
        # 분류 수행
        tag = classifier.classify_schedule_intent(text)
        
        print(f"분류 결과: {tag}")
        
        response_data = {
            'success': True,
            'input_text': text,
            'classified_tag': tag,
            'timestamp': datetime.now().isoformat(),
            'model': 'Qwen3',
            'confidence': 'high'
        }
        
        # UTF-8 응답
        response_json = json.dumps(response_data, ensure_ascii=False, indent=2)
        return Response(
            response_json,
            mimetype='application/json; charset=utf-8',
            status=200
        )
        
    except Exception as e:
        print(f"분류 오류: {str(e)}")
        
        error_response = {
            'success': False,
            'error': f'분류 오류: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }
        
        response_json = json.dumps(error_response, ensure_ascii=False, indent=2)
        return Response(
            response_json,
            mimetype='application/json; charset=utf-8',
            status=500
        )

@app.route('/health', methods=['GET'])
def health_check():
    status = "ok" if classifier.model is not None else "model_not_loaded"
    
    response_data = {
        'status': status,
        'message': 'Qwen3 분류 서버가 실행 중입니다',
        'model_loaded': classifier.model is not None,
        'device': classifier.device,
        'endpoints': {
            'classify': '/classify',
            'health': '/health'
        }
    }
    
    response_json = json.dumps(response_data, ensure_ascii=False, indent=2)
    return Response(
        response_json,
        mimetype='application/json; charset=utf-8'
    )

@app.route('/test', methods=['GET'])
def test_classification():
    """테스트용 엔드포인트"""
    test_cases = [
        "내일 3시에 회의 일정을 만들어줘",
        "다음주 화요일 미팅을 등록해줘", 
        "금요일 약속 시간을 4시로 수정해줘",
        "목요일 점심약속을 취소해줘"
    ]
    
    results = []
    for text in test_cases:
        tag = classifier.classify_schedule_intent(text)
        results.append({
            'text': text,
            'tag': tag
        })
    
    response_data = {
        'test_results': results,
        'total_tests': len(test_cases),
        'timestamp': datetime.now().isoformat()
    }
    
    response_json = json.dumps(response_data, ensure_ascii=False, indent=2)
    return Response(
        response_json,
        mimetype='application/json; charset=utf-8'
    )

if __name__ == '__main__':
    print("=" * 60)
    print("Qwen3 일정 분류 서버 시작")
    print("=" * 60)
    
    # 모델 로드
    if classifier.load_model():
        print("서버 준비 완료!")
        print(f"건강 체크: http://localhost:5001/health")
        print(f"테스트: http://localhost:5001/test")
        print(f"분류 API: http://localhost:5001/classify")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("❌ 모델 로딩 실패로 서버를 시작할 수 없습니다.")
        print("GPU 메모리 부족이거나 모델 다운로드 문제일 수 있습니다.")
        print("더 작은 모델이나 CPU 모드를 시도해보세요.")