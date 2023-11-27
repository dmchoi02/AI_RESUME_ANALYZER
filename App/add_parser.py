import PyPDF2
import spacy
import re
from nltk.tokenize import word_tokenize

class AddParser(object):
    def __init__(self, pdf_path):
        self.__details = {
            'blog': None,
            'github': None,
            'toeic' : None,
            'certificate' : None,
        }

        # PDF 파일을 읽어 텍스트 추출
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()

        # spaCy 모델 로드
        nlp = spacy.load("en_core_web_sm")
        # spaCy 문서 생성
        doc = nlp(text)
    
    

    # 문서를 탐색하여 URL 추출
        for token in doc:
            if "https://velog.io" in token.text:
                self.__details['blog'] = 'https://velog.io/' + token.text.split('https://velog.io/')[-1].split()[0]
            elif "https://blog.naver.com" in token.text:
                self.__details['blog'] = "https://blog.naver.com/" + token.text.split('https://blog.naver.com/')[-1].split()[0]
            elif "https://www.notion.so" in token.text:
                self.__details['blog'] = "https://www.notion.so/" + token.text.split('https://www.notion.so/')[-1].split()[0]
            elif "https://github.com" in token.text and "/" in token.text:
                # github_url = token.text.split('/')[-1]
                self.__details['github'] = "https://github.com/" + token.text.split('https://github.com/')[-1].split()[0]

        self.__details['toeic'] = find_first_number_after_word(text, 'toeic')
        self.__details['certificate'] = extract_entities_nltk(text)

    def get_extracted_data(self):
        return self.__details
    
# 토익
def find_first_number_after_word(text, target_word):
# 찾을 단어가 있는 위치를 찾음
    word_index = text.find(target_word)

    if word_index == -1:
        # 찾을 단어가 없으면 None 반환
        return None

    # 찾을 단어 이후의 문자열 추출
    substring = text[word_index + len(target_word):]

    # 정규 표현식을 사용하여 숫자를 찾음
    match = re.search(r'\b(?:[1-9]\d*|0)\b', substring)

    if match:
        # 찾은 숫자 반환
        return int(match.group())

    # 숫자를 찾지 못하면 None 반환
    return None
    


def extract_entities_nltk(text):
    # 단어로 토큰화
    tokens = word_tokenize(text)

    # 자격증 추출
    certifications = []

    is_certification = False
    certification_buffer = ""

    for word in tokens:
        if '자격증' in word:
            is_certification = True
        elif is_certification:
            # 띄어쓰기 있는 경우 합치기
            if certification_buffer and word.isalpha():
                certification_buffer += word
            elif word.isalnum():  # 알파벳 또는 숫자인 경우에만 추가
                certification_buffer += word + " "
            elif certification_buffer:  # 띄어쓰기 없는 경우
                certifications.append(certification_buffer.strip())
                certification_buffer = ""
        

    # '자격증' 였던 경우 certification_buffer 추가
    if certification_buffer:
        certifications.extend(certification_buffer.strip().split(' '))

    return certifications
    # 텍스트 예시 
    # sample_text = """
    # 자격증 : TOEIC, 정보처리기능사, 한국사3급, 컴퓨터활용능력2급
    # 출력 ['TOEIC', '정보처리기능사', '한국사3급', '컴퓨터활용능력2급']
    # 주의 : 띄어쓰기 금지
     
    
    
