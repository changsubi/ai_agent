from flask import Flask, request, jsonify
import whisper
import base64
import os
import tempfile
import subprocess
import json

app = Flask(__name__)

# UTF-8 설정
app.config['JSON_AS_ASCII'] = False  # 한글 직접 출력

print("Whisper 모델 로딩 중...")
model = whisper.load_model("base")
print("✅ 모델 로딩 완료!")

def convert_audio_format(input_path, output_path):
    """FFmpeg를 사용해 오디오 형식 변환"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return True, "변환 성공"
        else:
            return False, result.stderr
            
    except Exception as e:
        return False, str(e)

@app.route('/transcribe-base64', methods=['POST'])
def transcribe_base64():
    temp_dir = None
    
    try:
        data = request.json
        audio_base64 = data.get('audio_base64')
        filename = data.get('filename', 'audio_file')
        language = data.get('language', 'ko')
        
        if not audio_base64:
            return jsonify({'success': False, 'error': 'audio_base64가 필요합니다'}), 400
        
        print(f"파일명: {filename}")
        print(f"Base64 데이터 길이: {len(audio_base64)} 문자")
        
        # Base64 디코딩
        try:
            audio_content = base64.b64decode(audio_base64)
            print(f"디코딩된 파일 크기: {len(audio_content)} bytes")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Base64 디코딩 실패: {str(e)}'}), 400
        
        if len(audio_content) < 1000:
            return jsonify({
                'success': False,
                'error': f'파일이 너무 작습니다 ({len(audio_content)} bytes)'
            }), 400
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        
        file_ext = os.path.splitext(filename)[1].lower()
        if not file_ext:
            file_ext = '.mp3'
        
        original_file = os.path.join(temp_dir, f'original{file_ext}')
        converted_file = os.path.join(temp_dir, 'converted.wav')
        
        # 원본 파일 저장
        with open(original_file, 'wb') as f:
            f.write(audio_content)
        
        print(f"원본 파일 저장: {original_file}")
        
        # FFmpeg로 형식 변환
        print("오디오 형식 변환 중...")
        convert_success, convert_message = convert_audio_format(original_file, converted_file)
        
        if not convert_success:
            print(f"변환 실패, 원본 파일로 시도: {convert_message}")
            whisper_input = original_file
        else:
            print("변환 성공")
            whisper_input = converted_file
        
        # Whisper로 변환
        print(f"Whisper 변환 시작 (언어: {language})...")
        
        if language and language != 'auto':
            result = model.transcribe(whisper_input, language=language)
        else:
            result = model.transcribe(whisper_input)
        
        print("✅ 변환 완료!")
        print(f"감지된 언어: {result['language']}")
        print(f"텍스트: {result['text']}")
        
        # 응답 데이터 구성
        response_data = {
            'success': True,
            'filename': filename,
            'text': result['text'].strip(),
            'language': result['language'],
            'duration': round(result.get('duration', 0), 2),
            'word_count': len(result['text'].split()),
            'segments': [
                {
                    'start': round(seg['start'], 2),
                    'end': round(seg['end'], 2),
                    'text': seg['text'].strip()
                } for seg in result['segments'][:10]
            ]
        }
        
        # 한글 출력을 위한 직접 JSON 응답
        from flask import Response
        response_json = json.dumps(response_data, ensure_ascii=False, indent=2)
        
        return Response(
            response_json,
            mimetype='application/json; charset=utf-8',
            status=200
        )
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        
        error_response = {
            'success': False,
            'error': f'처리 에러: {str(e)}'
        }
        
        response_json = json.dumps(error_response, ensure_ascii=False, indent=2)
        return Response(
            response_json,
            mimetype='application/json; charset=utf-8',
            status=500
        )
        
    finally:
        # 임시 파일 정리
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print("임시 파일 정리 완료")
            except Exception as cleanup_error:
                print(f"임시 파일 정리 실패: {cleanup_error}")

@app.route('/health', methods=['GET'])
def health_check():
    response_data = {
        'status': 'ok',
        'message': 'Whisper API 서버가 실행 중입니다',
        'model': 'base',
        'encoding': 'UTF-8 지원',
        'endpoints': {
            'transcribe_base64': '/transcribe-base64',
            'health': '/health'
        }
    }
    
    response_json = json.dumps(response_data, ensure_ascii=False, indent=2)
    return Response(
        response_json,
        mimetype='application/json; charset=utf-8'
    )

if __name__ == '__main__':
    print("=" * 60)
    print("Whisper API 서버 시작 (UTF-8 지원)")
    print("=" * 60)
    print(f"건강 체크: http://localhost:5000/health")
    print(f"Base64 변환: http://localhost:5000/transcribe-base64")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)