# Developed by dnoobnerd [https://dnoobnerd.netlify.app]    Made with Streamlit
import nltk
nltk.download('stopwords')
#nltk.download('punkt')
from Imports import *


############################## 코스 추천 관련 ################################
# Core Pkg
import streamlit.components.v1 as stc 

# Load EDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 


# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat



# Recommendation Sys
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
	# indices of the course
	course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()
	# Index of course
	idx = course_indices[title]

	# Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_course_indices = [i[0] for i in sim_scores[1:]]
	selected_course_scores = [i[0] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_course_indices]
	result_df['similarity_score'] = selected_course_scores
	final_recommended_courses = result_df[['course_title','similarity_score','url','price','num_subscribers']]
	return final_recommended_courses.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>

</div>
"""

# Search For Course 
def search_term_if_not_found(term,df):
	result_df = df[df['course_title'].str.contains(term)]
	return result_df

####################################################################################################



# sql connector
#connection = pymysql.connect(host='localhost',user='root',password='root@MySQL4admin',db='cv')
connection = pymysql.connect(host='localhost',user='root',password='0000',db='mydb') #mysql과 연결
cursor = connection.cursor()

###### Setting Page Configuration (favicon, Logo, Title) ######
st.set_page_config(
   page_title="AI 이력서 분석기", #페이지 제목
   page_icon='./Logo/recommend.png', #페이지 로고
)


###### Main function run() ###### 
def run():
    
    # (Logo, Heading, Sidebar etc)
    
    img = Image.open('./Logo/RESUM.png')
    #img = Image.open('./APP/LOGO/RESUM.png')

    st.image(img)
    st.sidebar.markdown("# Choose Something...") 
    activities = ["사용자", "피드백", "소개", "관리자"] # 목록
    choice = st.sidebar.selectbox("주어진 옵션 중에서 선택하세요:", activities)
    link = '<b>Built with 🤍 by <a href="https://dnoobnerd.netlify.app/" style="text-decoration: none; color: #021659;">Deepak Padhi</a></b>' 
    st.sidebar.markdown(link, unsafe_allow_html=True) #방문자 수 표시 부분
    st.sidebar.markdown('''
        <!-- 사이트 방문자 -->

        <div id="sfct2xghr8ak6lfqt3kgru233378jya38dy" hidden></div>

        <noscript>
            <a href="https://www.freecounterstat.com" title="hit counter">
                <img src="https://counter9.stat.ovh/private/freecounterstat.php?c=t2xghr8ak6lfqt3kgru233378jya38dy" border="0" title="hit counter" alt="hit counter"> -->
            </a>
        </noscript>
    
        <p>Visitors <img src="https://counter9.stat.ovh/private/freecounterstat.php?c=t2xghr8ak6lfqt3kgru233378jya38dy" title="Free Counter" Alt="web counter" width="60px"  border="0" /></p>
    
    ''', unsafe_allow_html=True)

############################## 유데미 코스 데이터 ################################################################# 
    # df for Udemy Course recommendation 
    df = load_data("data/udemy_course_data.csv")
    result_df = pd.DataFrame(columns=['rec_title','rec_score','rec_url', 'rec_price', 'rec_num_sub'])
###############################################################################################################


    ###### Creating Database and Table ######
    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    # Create table user_data and user_feedback
    # 데이터베이스와 테이블을 생성하는 SQL쿼리 실행(웹페이지 출력 x)
    DB_table_name = 'user_data'
    # 토익, 깃헙주소, 블로그, 동아리, 자격증, 경력 데이터 추가
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    sec_token varchar(20) NOT NULL,
                    ip_add varchar(50) NULL,
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    toeic varchar(10) NULL, 
                    github_address varchar(100) NULL,
                    blog varchar(100) NULL,
                    club varchar(100) NULL,
                    certificate varchar(500) NULL,
                    PRIMARY KEY (ID)
                    );
                """
    cursor.execute(table_sql)

    

    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)
    ###### CODE FOR CLIENT SIDE (USER) ######

    

    if choice == '사용자': # 목록에서 사용자 선택한 경우
        
        # Collecting Miscellaneous Information
        act_name = st.text_input('이름*')
        act_mail = st.text_input('이메일*')
        act_mob  = st.text_input('휴대폰 번호*')
        
        

        if not act_name or not act_mail or not act_mob:
            st.warning("이름, 이메일 및 휴대폰 번호는 필수 입력 사항입니다. 모든 필수 정보를 입력해주세요.")
        else:
            sec_token = secrets.token_urlsafe(12)
            host_name = socket.gethostname()
            ip_add = socket.gethostbyname(host_name)
            dev_user = os.getlogin()
            os_name_ver = platform.system() + " " + platform.release()
            g = geocoder.ip('me')
            latlong = g.latlng #geolocator 모듈을 이용해 ip로 지리위치정보 획득
            geolocator = Nominatim(user_agent="http")
            location = geolocator.reverse(latlong, language='kr')
            address = location.raw['address']
            cityy = address.get('city', '')
            statee = address.get('state', '')
            countryy = address.get('country', '')  
            city = cityy
            state = statee
            country = countryy


            # Upload Resume
            st.markdown('''<h5 style='text-align: left; color: #021659;'> 이력서를 업로드하고 스마트한 추천을 받아보세요</h5>''',unsafe_allow_html=True)
            
            ## file upload in pdf format
            pdf_file = st.file_uploader("이력서를 선택하세요.", type=["pdf"])
            if pdf_file is not None:
                with st.spinner('잠시만 기다려주세요...'):
                    time.sleep(4)
            
                ### saving the uploaded resume to folder
                save_image_path = './Uploaded_Resumes/'+pdf_file.name
                pdf_name = pdf_file.name
                with open(save_image_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                show_pdf(save_image_path)

                ### parsing and extracting whole resume 
                resume_data = ResumeParser(save_image_path).get_extracted_data()
                resume_url_data = AddParser(save_image_path).get_extracted_data()
                github_address = resume_url_data['github']
                blog = resume_url_data['blog']
                toeic = resume_url_data['toeic']
                certificate = resume_url_data['certificate'] 
                club = 'club'

                if resume_data:
                    
                    ## Get the whole resume data into resume_text
                    resume_text = pdf_reader(save_image_path)

                    ## Showing Analyzed data from (resume_data)
                    st.header("**이력서 분석 🤘**")
                    st.success("안녕하세요 "+ act_name + "님")
                    st.subheader("**기본 정보 👀**")
                    try:
                        st.text('이름: '+ act_name)
                        st.text('이메일: ' + act_mail)
                        st.text('연락처: ' + act_mob)                 
                        st.text('이력서 페이지 수: '+str(resume_data['no_of_pages']))

                    except:
                        pass
                    ## Predicting Candidate Experience Level 

                    ### Trying with different possibilities
                    cand_level = ''
                    if resume_data['no_of_pages'] < 1:                
                        cand_level = "NA"
                        st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>당신은 신입 수준입니다!</h4>''',unsafe_allow_html=True)
                    
                    #### if internship then intermediate level
                    #이력서가 한글이라 조건도 한글이 되어야 하나해서 일단 번역했습니다
                    elif '인턴십' in resume_text:
                        cand_level = "중급"
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>당신은 중급 수준입니다!</h4>''',unsafe_allow_html=True)
                    elif '인턴쉽' in resume_text:
                        cand_level = "중급"
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>당신은 중급 수준입니다!</h4>''',unsafe_allow_html=True)
                    elif 'Internship' in resume_text:
                        cand_level = "중급"
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>당신은 중급 수준입니다!</h4>''',unsafe_allow_html=True)
                    elif 'Internships' in resume_text:
                        cand_level = "중급"
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>당신은 중급 수준입니다!</h4>''',unsafe_allow_html=True)
                    
                    #### if Work Experience/Experience then Experience level
                    elif '경력' in resume_text:
                        cand_level = "경험자"
                        st.markdown('''<h4 style='text-align: left; color: #fba171;'>당신은 경력자 수준입니다!''',unsafe_allow_html=True)
                    elif 'WORK EXPERIENCE' in resume_text:
                        cand_level = "경험자"
                        st.markdown('''<h4 style='text-align: left; color: #fba171;'>당신은 경력자 수준입니다!''',unsafe_allow_html=True)
                    elif 'Experience' in resume_text:
                        cand_level = "경험자"
                        st.markdown('''<h4 style='text-align: left; color: #fba171;'>당신은 경력자 수준입니다!''',unsafe_allow_html=True)
                    elif 'Work Experience' in resume_text:
                        cand_level = "경험자"
                        st.markdown('''<h4 style='text-align: left; color: #fba171;'>당신은 경력자 수준입니다!''',unsafe_allow_html=True)
                    else:
                        cand_level = "신입"
                        st.markdown('''<h4 style='text-align: left; color: #fba171;'>당신은 신입 수준입니다!''',unsafe_allow_html=True)


                    ## Skills Analyzing and Recommendation
                    st.subheader("**기술 추천 💡**")
                    
                    ### Current Analyzed Skills
                    keywords = st_tags(label='### 현재 보유한 기술',
                    text='아래에서 기술 추천을 확인하세요',value=resume_data['skills'],key = '1  ')

                    ### Keywords for Recommendations
                    ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                    web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'C#', 'Asp.net', 'flask']
                    android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                    ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                    uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','입체','파악','사용자 조사','사용자 경험']
                    n_any = ['영어','의사소통','글쓰기', 'microsoft office 작업', '리더십','고객관리', '소셜 미디어']
                    ### Skill Recommendations Starts                
                    recommended_skills = []
                    reco_field = ''
                    rec_course = ''

                ########################################################### 추가/수정한 부분 #################################################################

                    ### condition starts to check skills from keywords and predict field
                    for i in resume_data['skills']:
                    
                        #### Data science recommendation
                        if i.lower() in ds_keyword:
                            print(i.lower())
                            reco_field = '데이터 과학'
                            st.success("** 분석 결과 데이터 과학 직종을 탐색 중이라고 판단됩니다.**")
                            recommended_skills = ['데이터 시각화','예측 분석','통계 모델링','데이터 마이닝','클러스터링 및 분류','데이터 분석','양적 분석','웹 스크래핑','머신 러닝 알고리즘','Keras','Pytorch','확률','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                            recommended_keywords = st_tags(label='### 당신을 위한 추천 기술.',
                            text='시스템에서 생성된 추천 기술',value=recommended_skills,key = '2')
                            st.markdown('''<h5 style='text-align: left; color: #1ed760;'>이러한 기술을 이력서에 추가하면 취업 기회가 향상될 것 입니다.🚀 </h5>''',unsafe_allow_html=True)
                            # course recommendation
                            rec_course = course_recommender(ds_course)

                            # Udemy recommendation
                            st.subheader("Recommended Udemy Courses")
                            cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
                            num_of_rec = st.slider("Choose Number of Course Recommendations:",3,30,5)
                            search_terms = ["Data Visualization", "Flask", "Analysis", "Modeling", "Data Analytics"]
                            
                            try:
                                results = get_recommendation(search_terms[0],cosine_sim_mat,df,num_of_rec)

                                for row in results.iterrows():
                                    rec_title = row[1][0]
                                    rec_score = row[1][1]
                                    rec_url = row[1][2]
                                    rec_price = row[1][3]
                                    rec_num_sub = row[1][4]
                                
                                
                                result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                result_df['rec_score'] = pd.Series(rec_score).reset_index(drop=True)
                                result_df['rec_url'] = pd.Series(rec_url).reset_index(drop=True)
                                result_df['rec_price'] = pd.Series(rec_price).reset_index(drop=True)
                                result_df['rec_num_sub'] = pd.Series(rec_num_sub).reset_index(drop=True)


                            except:
                                except_df = search_term_if_not_found(search_terms[0],df)

                                result_df['rec_title'] = pd.concat([result_df['rec_title'], except_df['course_title']], ignore_index=True)
                                result_df['rec_score'] = pd.Series([random.random() for _ in range(except_df.shape[0])])
                                result_df['rec_url'] = except_df['url'].reset_index(drop=True)
                                result_df['rec_price'] = except_df['price'].reset_index(drop=True)
                                result_df['rec_num_sub'] = except_df['num_subscribers'].reset_index(drop=True)

                            
                            for i in range(1, len(search_terms)):
                                search_term = search_terms[i]

                                try:
                                    results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)

                                    for row in results.iterrows():
                                        rec_title = row[1][0]
                                        rec_score = row[1][1]
                                        rec_url = row[1][2]
                                        rec_price = row[1][3]
                                        rec_num_sub = row[1][4]
                                    
                                    result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                    result_df['rec_score'] = pd.concat([result_df['rec_score'], pd.Series(rec_score)], ignore_index=True)
                                    result_df['rec_url'] = pd.concat([result_df['rec_url'], pd.Series(rec_url)], ignore_index=True)
                                    result_df['rec_price'] = pd.concat([result_df['rec_price'], pd.Series(rec_price)], ignore_index=True)
                                    result_df['rec_num_sub'] = pd.concat([result_df['rec_num_sub'], pd.Series(rec_num_sub)], ignore_index=True)


                                except:
                                    except_df = search_term_if_not_found(search_term,df)
                                    except_df = except_df.reset_index(drop=True)
                                    result_df = pd.concat([result_df, pd.DataFrame({'rec_title': except_df['course_title'],'rec_score': pd.Series([random.random() for _ in range(except_df.shape[0])]), 'rec_url': except_df['url'], 'rec_price': except_df['price'], 'rec_num_sub': except_df['num_subscribers']})], ignore_index=True)
                            
                            
                            result_df = result_df.sample(frac=1).reset_index(drop=True)

                            for i in range(num_of_rec):       
                                stc.html(RESULT_TEMP.format(result_df['rec_title'].values[i],result_df['rec_score'].values[i],result_df['rec_url'].values[i],result_df['rec_price'].values[i],result_df['rec_num_sub'].values[i]),height=350)

                            break
                        
                        #### Web development recommendation
                        elif i.lower() in web_keyword:
                            print(i.lower())
                            reco_field = '웹 개발'
                            st.success("**분석 결과 웹 개발 직종을 탐색 중이라고 판단됩니다. **")
                            recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                            recommended_keywords = st_tags(label='### 당신을 위한 추천 기술.',
                            text='시스템에서 생성된 추천 기술',value=recommended_skills,key = '3')
                            st.markdown('''<h5 style='text-align: left; color: #1ed760;'>이러한 기술을 이력서에 추가하면 취업 기회가 향상될 것입니다🚀💼</h5>''',unsafe_allow_html=True)
                            # course recommendation
                            rec_course = course_recommender(web_course)

                            # Udemy recommendation
                            st.subheader("Recommended Udemy Courses")
                            cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
                            num_of_rec = st.slider("Choose Number of Course Recommendations:",3,30,5)
                            search_terms = ["React", "Django", "Node.js", "Javascript", "php"]
                            
                            try:
                                results = get_recommendation(search_terms[0],cosine_sim_mat,df,num_of_rec)

                                for row in results.iterrows():
                                    rec_title = row[1][0]
                                    rec_score = row[1][1]
                                    rec_url = row[1][2]
                                    rec_price = row[1][3]
                                    rec_num_sub = row[1][4]
                                
                                
                                result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                result_df['rec_score'] = pd.Series(rec_score).reset_index(drop=True)
                                result_df['rec_url'] = pd.Series(rec_url).reset_index(drop=True)
                                result_df['rec_price'] = pd.Series(rec_price).reset_index(drop=True)
                                result_df['rec_num_sub'] = pd.Series(rec_num_sub).reset_index(drop=True)


                            except:
                                except_df = search_term_if_not_found(search_terms[0],df)

                                result_df['rec_title'] = pd.concat([result_df['rec_title'], except_df['course_title']], ignore_index=True)
                                result_df['rec_score'] = pd.Series([random.random() for _ in range(except_df.shape[0])])
                                result_df['rec_url'] = except_df['url'].reset_index(drop=True)
                                result_df['rec_price'] = except_df['price'].reset_index(drop=True)
                                result_df['rec_num_sub'] = except_df['num_subscribers'].reset_index(drop=True)

                            
                            for i in range(1, len(search_terms)):
                                search_term = search_terms[i]

                                try:
                                    results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)

                                    for row in results.iterrows():
                                        rec_title = row[1][0]
                                        rec_score = row[1][1]
                                        rec_url = row[1][2]
                                        rec_price = row[1][3]
                                        rec_num_sub = row[1][4]
                                    
                                    result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                    result_df['rec_score'] = pd.concat([result_df['rec_score'], pd.Series(rec_score)], ignore_index=True)
                                    result_df['rec_url'] = pd.concat([result_df['rec_url'], pd.Series(rec_url)], ignore_index=True)
                                    result_df['rec_price'] = pd.concat([result_df['rec_price'], pd.Series(rec_price)], ignore_index=True)
                                    result_df['rec_num_sub'] = pd.concat([result_df['rec_num_sub'], pd.Series(rec_num_sub)], ignore_index=True)


                                except:
                                    except_df = search_term_if_not_found(search_term,df)
                                    except_df = except_df.reset_index(drop=True)
                                    result_df = pd.concat([result_df, pd.DataFrame({'rec_title': except_df['course_title'],'rec_score': pd.Series([random.random() for _ in range(except_df.shape[0])]), 'rec_url': except_df['url'], 'rec_price': except_df['price'], 'rec_num_sub': except_df['num_subscribers']})], ignore_index=True)
                            
                            
                            result_df = result_df.sample(frac=1).reset_index(drop=True)

                            for i in range(num_of_rec):       
                                stc.html(RESULT_TEMP.format(result_df['rec_title'].values[i],result_df['rec_score'].values[i],result_df['rec_url'].values[i],result_df['rec_price'].values[i],result_df['rec_num_sub'].values[i]),height=350)

                            break

                        #### Android App Development
                        elif i.lower() in android_keyword:
                            print(i.lower())
                            reco_field = '안드로이드 앱 개발'
                            st.success("** 분석 결과 안드로이드 앱 개발 직종을 탐색 중이라고 판단됩니다. **")
                            recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                            recommended_keywords = st_tags(label='### 당신을 위한 추천 기술',
                            text='시스템에서 생성된 추천 기술',value=recommended_skills,key = '4')
                            st.markdown('''<h5 style='text-align: left; color: #1ed760;'>이러한 기술을 이력서에 추가하면 취업 기회가 향상될 것입니다🚀💼</h5>''',unsafe_allow_html=True)
                            # course recommendation
                            rec_course = course_recommender(android_course)
                            
                            # Udemy recommendation
                            st.subheader("Recommended Udemy Courses")
                            cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
                            num_of_rec = st.slider("Choose Number of Course Recommendations:",3,30,5)
                            search_terms = ["Android", "XML", "Java", "SQL", "Javascript"]
                            
                            try:
                                results = get_recommendation(search_terms[0],cosine_sim_mat,df,num_of_rec)

                                for row in results.iterrows():
                                    rec_title = row[1][0]
                                    rec_score = row[1][1]
                                    rec_url = row[1][2]
                                    rec_price = row[1][3]
                                    rec_num_sub = row[1][4]
                                
                                
                                result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                result_df['rec_score'] = pd.Series(rec_score).reset_index(drop=True)
                                result_df['rec_url'] = pd.Series(rec_url).reset_index(drop=True)
                                result_df['rec_price'] = pd.Series(rec_price).reset_index(drop=True)
                                result_df['rec_num_sub'] = pd.Series(rec_num_sub).reset_index(drop=True)


                            except:
                                except_df = search_term_if_not_found(search_terms[0],df)

                                result_df['rec_title'] = pd.concat([result_df['rec_title'], except_df['course_title']], ignore_index=True)
                                result_df['rec_score'] = pd.Series([random.random() for _ in range(except_df.shape[0])])
                                result_df['rec_url'] = except_df['url'].reset_index(drop=True)
                                result_df['rec_price'] = except_df['price'].reset_index(drop=True)
                                result_df['rec_num_sub'] = except_df['num_subscribers'].reset_index(drop=True)

                            
                            for i in range(1, len(search_terms)):
                                search_term = search_terms[i]

                                try:
                                    results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)

                                    for row in results.iterrows():
                                        rec_title = row[1][0]
                                        rec_score = row[1][1]
                                        rec_url = row[1][2]
                                        rec_price = row[1][3]
                                        rec_num_sub = row[1][4]
                                    
                                    result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                    result_df['rec_score'] = pd.concat([result_df['rec_score'], pd.Series(rec_score)], ignore_index=True)
                                    result_df['rec_url'] = pd.concat([result_df['rec_url'], pd.Series(rec_url)], ignore_index=True)
                                    result_df['rec_price'] = pd.concat([result_df['rec_price'], pd.Series(rec_price)], ignore_index=True)
                                    result_df['rec_num_sub'] = pd.concat([result_df['rec_num_sub'], pd.Series(rec_num_sub)], ignore_index=True)


                                except:
                                    except_df = search_term_if_not_found(search_term,df)
                                    except_df = except_df.reset_index(drop=True)
                                    result_df = pd.concat([result_df, pd.DataFrame({'rec_title': except_df['course_title'],'rec_score': pd.Series([random.random() for _ in range(except_df.shape[0])]), 'rec_url': except_df['url'], 'rec_price': except_df['price'], 'rec_num_sub': except_df['num_subscribers']})], ignore_index=True)
                            
                            
                            result_df = result_df.sample(frac=1).reset_index(drop=True)

                            for i in range(num_of_rec):       
                                stc.html(RESULT_TEMP.format(result_df['rec_title'].values[i],result_df['rec_score'].values[i],result_df['rec_url'].values[i],result_df['rec_price'].values[i],result_df['rec_num_sub'].values[i]),height=350)

                            break

                        #### IOS App Development
                        elif i.lower() in ios_keyword:
                            print(i.lower())
                            reco_field = 'IOS 앱 개발'
                            st.success("**분석 결과 iOS 앱 개발 직종을 탐색 중이라고 판단됩니다. **")
                            recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                            recommended_keywords = st_tags(label='### 당신을 위한 추천 기술',
                            text='시스템에서 생성된 추천 기술',value=recommended_skills,key = '5')
                            st.markdown('''<h5 style='text-align: left; color: #1ed760;'>이러한 기술을 이력서에 추가하면 취업 기회가 향상될 것입니다🚀💼</h5>''',unsafe_allow_html=True)
                            # course recommendation
                            rec_course = course_recommender(ios_course)
                            
                            # Udemy recommendation
                            st.subheader("Recommended Udemy Courses")
                            cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
                            num_of_rec = st.slider("Choose Number of Course Recommendations:",3,30,5)
                            search_terms = ["IOS", "Swift", "SQL", "Firebase", "git"]
                            
                            try:
                                results = get_recommendation(search_terms[0],cosine_sim_mat,df,num_of_rec)

                                for row in results.iterrows():
                                    rec_title = row[1][0]
                                    rec_score = row[1][1]
                                    rec_url = row[1][2]
                                    rec_price = row[1][3]
                                    rec_num_sub = row[1][4]
                                
                                
                                result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                result_df['rec_score'] = pd.Series(rec_score).reset_index(drop=True)
                                result_df['rec_url'] = pd.Series(rec_url).reset_index(drop=True)
                                result_df['rec_price'] = pd.Series(rec_price).reset_index(drop=True)
                                result_df['rec_num_sub'] = pd.Series(rec_num_sub).reset_index(drop=True)


                            except:
                                except_df = search_term_if_not_found(search_terms[0],df)

                                result_df['rec_title'] = pd.concat([result_df['rec_title'], except_df['course_title']], ignore_index=True)
                                result_df['rec_score'] = pd.Series([random.random() for _ in range(except_df.shape[0])])
                                result_df['rec_url'] = except_df['url'].reset_index(drop=True)
                                result_df['rec_price'] = except_df['price'].reset_index(drop=True)
                                result_df['rec_num_sub'] = except_df['num_subscribers'].reset_index(drop=True)

                            
                            for i in range(1, len(search_terms)):
                                search_term = search_terms[i]

                                try:
                                    results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)

                                    for row in results.iterrows():
                                        rec_title = row[1][0]
                                        rec_score = row[1][1]
                                        rec_url = row[1][2]
                                        rec_price = row[1][3]
                                        rec_num_sub = row[1][4]
                                    
                                    result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                    result_df['rec_score'] = pd.concat([result_df['rec_score'], pd.Series(rec_score)], ignore_index=True)
                                    result_df['rec_url'] = pd.concat([result_df['rec_url'], pd.Series(rec_url)], ignore_index=True)
                                    result_df['rec_price'] = pd.concat([result_df['rec_price'], pd.Series(rec_price)], ignore_index=True)
                                    result_df['rec_num_sub'] = pd.concat([result_df['rec_num_sub'], pd.Series(rec_num_sub)], ignore_index=True)


                                except:
                                    except_df = search_term_if_not_found(search_term,df)
                                    except_df = except_df.reset_index(drop=True)
                                    result_df = pd.concat([result_df, pd.DataFrame({'rec_title': except_df['course_title'],'rec_score': pd.Series([random.random() for _ in range(except_df.shape[0])]), 'rec_url': except_df['url'], 'rec_price': except_df['price'], 'rec_num_sub': except_df['num_subscribers']})], ignore_index=True)
                            
                            
                            result_df = result_df.sample(frac=1).reset_index(drop=True)

                            for i in range(num_of_rec):       
                                stc.html(RESULT_TEMP.format(result_df['rec_title'].values[i],result_df['rec_score'].values[i],result_df['rec_url'].values[i],result_df['rec_price'].values[i],result_df['rec_num_sub'].values[i]),height=350)

                            break

                        #### Ui-UX Recommendation
                        elif i.lower() in uiux_keyword:
                            print(i.lower())
                            reco_field = 'UI-UX 개발'
                            st.success("** 분석 결과 UI-UX 개발 직종을 탐색 중이라고 판단됩니다. **")
                            recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                            recommended_keywords = st_tags(label='### 당신을 위한 추천 기술 ',
                            text='시스템에서 생성된 추천 기술',value=recommended_skills,key = '6')
                            st.markdown('''<h5 style='text-align: left; color: #1ed760;'>이러한 기술을 이력서에 추가하면 취업 기회가 향상될 것입니다🚀💼</h5>''',unsafe_allow_html=True)
                            # course recommendation
                            rec_course = course_recommender(uiux_course)
                            
                            # Udemy recommendation
                            st.subheader("Recommended Udemy Courses")
                            cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
                            num_of_rec = st.slider("Choose Number of Course Recommendations:",3,30,5)
                            search_terms = ["UI", "Adobe", "UX", "Illustrator", "Editing"]
                            
                            try:
                                results = get_recommendation(search_terms[0],cosine_sim_mat,df,num_of_rec)

                                for row in results.iterrows():
                                    rec_title = row[1][0]
                                    rec_score = row[1][1]
                                    rec_url = row[1][2]
                                    rec_price = row[1][3]
                                    rec_num_sub = row[1][4]
                                
                                
                                result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                result_df['rec_score'] = pd.Series(rec_score).reset_index(drop=True)
                                result_df['rec_url'] = pd.Series(rec_url).reset_index(drop=True)
                                result_df['rec_price'] = pd.Series(rec_price).reset_index(drop=True)
                                result_df['rec_num_sub'] = pd.Series(rec_num_sub).reset_index(drop=True)


                            except:
                                except_df = search_term_if_not_found(search_terms[0],df)

                                result_df['rec_title'] = pd.concat([result_df['rec_title'], except_df['course_title']], ignore_index=True)
                                result_df['rec_score'] = pd.Series([random.random() for _ in range(except_df.shape[0])])
                                result_df['rec_url'] = except_df['url'].reset_index(drop=True)
                                result_df['rec_price'] = except_df['price'].reset_index(drop=True)
                                result_df['rec_num_sub'] = except_df['num_subscribers'].reset_index(drop=True)

                            
                            for i in range(1, len(search_terms)):
                                search_term = search_terms[i]

                                try:
                                    results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)

                                    for row in results.iterrows():
                                        rec_title = row[1][0]
                                        rec_score = row[1][1]
                                        rec_url = row[1][2]
                                        rec_price = row[1][3]
                                        rec_num_sub = row[1][4]
                                    
                                    result_df['rec_title'] = pd.concat([result_df['rec_title'], pd.Series(rec_title)], ignore_index=True)
                                    result_df['rec_score'] = pd.concat([result_df['rec_score'], pd.Series(rec_score)], ignore_index=True)
                                    result_df['rec_url'] = pd.concat([result_df['rec_url'], pd.Series(rec_url)], ignore_index=True)
                                    result_df['rec_price'] = pd.concat([result_df['rec_price'], pd.Series(rec_price)], ignore_index=True)
                                    result_df['rec_num_sub'] = pd.concat([result_df['rec_num_sub'], pd.Series(rec_num_sub)], ignore_index=True)


                                except:
                                    except_df = search_term_if_not_found(search_term,df)
                                    except_df = except_df.reset_index(drop=True)
                                    result_df = pd.concat([result_df, pd.DataFrame({'rec_title': except_df['course_title'],'rec_score': pd.Series([random.random() for _ in range(except_df.shape[0])]), 'rec_url': except_df['url'], 'rec_price': except_df['price'], 'rec_num_sub': except_df['num_subscribers']})], ignore_index=True)
                            
                            
                            result_df = result_df.sample(frac=1).reset_index(drop=True)

                            for i in range(num_of_rec):       
                                stc.html(RESULT_TEMP.format(result_df['rec_title'].values[i],result_df['rec_score'].values[i],result_df['rec_url'].values[i],result_df['rec_price'].values[i],result_df['rec_num_sub'].values[i]),height=350)

                            break

                        #### For Not Any Recommendations
                        elif i.lower() in n_any:
                            print(i.lower())
                            reco_field = 'NA'
                            st.warning("** 현재 우리 도구는 데이터 과학, 웹, 안드로이드, iOS 및 UI/UX 개발에 대해서만 예측 및 추천을 제공합니다. **")
                            recommended_skills = ['추천 없음']
                            recommended_keywords = st_tags(label='### 당신을 위한 추천 기술',
                            text='현재 추천 사항이 없습니다. ',value=recommended_skills,key = '6')
                            st.markdown('''<h5 style='text-align: left; color: #092851;'>향후 업데이트에서 추가될 수 있습니다</h5>''',unsafe_allow_html=True)
                            # course recommendation
                            rec_course = "S죄송합니다! 이 분야에 대한 추천이 현재 불가능합니다. "
                            break


                    ## Resume Scorer & Resume Writing Tips
                    st.subheader("**이력서 작성 팁 & 아이디어 🥂**")
                    resume_score = 0
                    
                    ### Predicting Whether these key points are added to the resume
                    if '블로그' or 'blog' or '블로그' in resume_text:
                        resume_score = resume_score+6
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 기술 블로그가 추가되었습니다.</h4>''',unsafe_allow_html=True)                
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 기술 블로그를 추가해 주세요. 이것은 채용 담당자에게 귀하의 기술 성장 과정을 전달할 수 있습니다.</h4>''',unsafe_allow_html=True)

                    if '교육' or '학교' or '대학'  in resume_text:
                        resume_score = resume_score + 12
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 교육 세부 정보가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 교육 세부 정보를 추가해 주세요. 이것은 채용 담당자에게 귀하의 자격 수준을 알려줄 것입니다.</h4>''',unsafe_allow_html=True)

                    if '경력' in resume_text:
                        resume_score = resume_score + 16
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 경력이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Experience' in resume_text:
                        resume_score = resume_score + 16
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 경력이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 경력을 추가해 주세요. 이것은 다른 지원자들과 차별화될 수 있습니다.</h4>''',unsafe_allow_html=True)

                    if '인턴십'  in resume_text:
                        resume_score = resume_score + 6
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 인턴십이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif '인턴쉽'  in resume_text:
                        resume_score = resume_score + 6
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 인턴쉽이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Internships'  in resume_text:
                        resume_score = resume_score + 6
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 인턴쉽이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Internship'  in resume_text:
                        resume_score = resume_score + 6
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 인턴쉽이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 인턴쉽을 추가해 주세요. 이것은 다른 지원자들과 차별화될 수 있습니다.</h4>''',unsafe_allow_html=True)

                    if '기술'  in resume_text:
                        resume_score = resume_score + 11
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 기술이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'SKILL'  in resume_text:
                        resume_score = resume_score + 11
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 기술이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Skills'  in resume_text:
                        resume_score = resume_score + 11
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 기술이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Skill'  in resume_text:
                        resume_score = resume_score + 11
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 기술이 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 기술을 추가해 주세요. 이것은 여러분을 도울 매우 중요한 정보입니다.</h4>''',unsafe_allow_html=True)

                    if '취미' or '특기' in resume_text:
                        resume_score = resume_score + 4
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 취미가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Hobbies' in resume_text:
                        resume_score = resume_score + 4
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 취미가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 취미를 추가해 주세요. 이것은 여러분의 성격을 채용 담당자에게 보여주고, 이 역할에 적합한지 여부를 보여줄 것입니다.</h4>''',unsafe_allow_html=True)

                    if '관심사' in resume_text:
                        resume_score = resume_score + 5
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 관심사가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Interests'in resume_text:
                        resume_score = resume_score + 5
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 관심사가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 관심사를 추가해 주세요. 이것은 채용 담당자에게 여러분의 업무 외 관심사를 보여줄 수 있습니다.</h4>''',unsafe_allow_html=True)

                    if '성취' or '성과' in resume_text:
                        resume_score = resume_score + 6
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 성취가 추가되었습니다. </h4>''',unsafe_allow_html=True)
                    elif 'Achievements' in resume_text:
                        resume_score = resume_score + 6
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 성취가 추가되었습니다. </h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 성취를 추가해 주세요. 이것은 여러분이 필요한 업무에 적합한 능력을 나타낼 수 있습니다.</h4>''',unsafe_allow_html=True)

                    if '자격증' in resume_text:
                        resume_score = resume_score + 8
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 자격증이 추가되었습니다. </h4>''',unsafe_allow_html=True)
                    elif 'Certifications' in resume_text:
                        resume_score = resume_score + 8
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 자격증이 추가되었습니다. </h4>''',unsafe_allow_html=True)
                    elif 'Certification' in resume_text:
                        resume_score = resume_score + 8
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 자격증이 추가되었습니다. </h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 자격증을 추가해 주세요. 이것은 여러분이 필요한 업무에 대해 전문 지식을 갖고 있다는 것을 보여줄 수 있습니다.</h4>''',unsafe_allow_html=True)

                    if '프로젝트' in resume_text:
                        resume_score = resume_score + 26
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 프로젝트가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'PROJECT' in resume_text:
                        resume_score = resume_score + 26
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 프로젝트가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Projects' in resume_text:
                        resume_score = resume_score + 26
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 프로젝트가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    elif 'Project' in resume_text:
                        resume_score = resume_score + 26
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] 훌륭합니다! 프로젝트가 추가되었습니다.</h4>''',unsafe_allow_html=True)
                    else:
                        st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] 프로젝트를 추가해 주세요. 이것은 여러분이 필요한 업무와 관련된 작업을 수행했는지 여부를 나타낼 수 있습니다.</h4>''',unsafe_allow_html=True)

                    st.subheader("**이력서 점수 📝**")
                    
                    st.markdown(
                        """
                        <style>
                            .stProgress > div > div > div > div {
                                background-color: #d73b5c;
                            }
                        </style>""",
                        unsafe_allow_html=True,
                    )

                    ### Score Bar
                    my_bar = st.progress(0)
                    score = 0
                    for percent_complete in range(resume_score):
                        score +=1
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 1)

                    ### Score
                    st.success('** 이력서 작성 점수: ' + str(score)+'**')
                    st.warning("** 참고: 이 점수는 이력서 내용을 기반으로 계산됩니다. **")

                    # print(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)


                    ### Getting Current Date and Time
                    ts = time.time()
                    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    timestamp = str(cur_date+'_'+cur_time)


                    ## Calling insert_data to add all the data into user_data                
                    insert_data(cursor, connection, str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name, str(toeic), str(github_address), str(blog), str(club), str(certificate))

                    ## Recommending Resume Writing Video
                    #st.header("**Bonus Video for Resume Writing Tips💡**")
                    st.header("**이력서 작성을 위한 보너스 영상💡**")
                    resume_vid = random.choice(resume_videos) # 랜덤으로 선택
                    st.video(resume_vid)

                    ## Recommending Interview Preparation Video
                    #st.header("**Bonus Video for Interview Tips💡**")
                    st.header("**면접을 위한 보너스 영상💡**")
                    interview_vid = random.choice(interview_videos) # 랜덤으로 선택
                    st.video(interview_vid)

                    ## On Successful Result 
              
                # st.balloons()
                else:
                    st.error('문제가 발생했습니다...')                


    ###### CODE FOR FEEDBACK SIDE ######
    elif choice == '피드백':   
        
        # timestamp 
        ts = time.time()
        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        timestamp = str(cur_date+'_'+cur_time)

        # Feedback Form
        with st.form("my_form"):
            st.write("피드백 양식")            
            feed_name = st.text_input('이름')
            feed_email = st.text_input('이메일')
            feed_score = st.slider('점수를 메겨주세요. (1 에서 5)', 1, 5)
            comments = st.text_input('의견')
            Timestamp = timestamp        
            submitted = st.form_submit_button("제출")
            if submitted:
                ## Calling insertf_data to add dat into user feedback
                insertf_data(cursor, connection, feed_name,feed_email,feed_score,comments,Timestamp)    
                ## Success Message 
                st.success("감사합니다! 당신의 피드백이 기록되었습니다.") 
                ## On Successful Submit
                st.balloons()    


        # query to fetch data from user feedback table
        query = 'select * from user_feedback'        
        plotfeed_data = pd.read_sql(query, connection)                        


        # fetching feed_score from the query and getting the unique values and total value count 
        labels = plotfeed_data.feed_score.unique()
        values = plotfeed_data.feed_score.value_counts()


        # plotting pie chart for user ratings
        st.subheader("**과거 사용자 등급**")
        fig = px.pie(values=values, names=labels, title="1에서 5까지의 사용자 등급 차트", color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig)


        #  Fetching Comment History
        cursor.execute('select feed_name, comments from user_feedback')
        plfeed_cmt_data = cursor.fetchall()

        st.subheader("**사용자 댓글**")
        dff = pd.DataFrame(plfeed_cmt_data, columns=['사용자', '댓글'])
        st.dataframe(dff, width=1000)

    
    ###### CODE FOR ABOUT PAGE ######
    elif choice == '소개':   

        st.subheader("**툴 소개 - AI 이력서 분석기**")

        st.markdown('''

        <p align='justify'>
            자연어 처리를 사용하여 이력서에서 정보를 추출하고 키워드를 찾아 그들을 키워드에 기반하여 섹터로 클러스터링합니다. 마지막으로 키워드 매칭을 기반으로 지원자에게 권장 사항, 예측 및 분석을 표시합니다.
        </p>

        <p align="justify">
            <b>사용 방법: -</b> <br/><br/>
            <b>사용자 -</b> <br/>
            사이드 바에서 사용자로 자신을 선택하고 필수 필드를 작성하고 이력서를 PDF 형식으로 업로드하십시오.<br/>
            그냥 앉아서 기다리세요. 우리의 도구가 스스로 마법을 부립니다.<br/><br/>
            <b>피드백 -</b> <br/>
            사용자가 도구에 대한 의견을 제안할 수 있는 곳입니다.<br/><br/>
            <b>관리자 -</b> <br/>
            로그인에는 사용자 이름으로 <b>admin</b> 및 비밀번호로 <b>admin@resume-analyzer</b>를 사용하십시오.<br/>
            필요한 모든 것을로드하고 분석을 수행합니다.
        </p><br/><br/>

        <p align="justify">
            Built with 🤍 by 
            <a href="https://dnoobnerd.netlify.app/" style="text-decoration: none; color: grey;">Deepak Padhi</a> through 
            <a href="https://www.linkedin.com/in/mrbriit/" style="text-decoration: none; color: grey;">Dr Bright --(Data Scientist)</a>
        </p>

        ''',unsafe_allow_html=True)  


    ###### CODE FOR ADMIN SIDE (ADMIN) ######
    else:
        st.success('관리자 페이지에 오신 것을 환영합니다.')

        #  Admin Login
        ad_user = st.text_input("사용자 이름")
        ad_password = st.text_input("비밀번호", type='password')

        if st.button('로그인'):
            
            ## Credentials 
            if ad_user == 'admin' and ad_password == 'admin@resume-analyzer':
                
                ### Fetch miscellaneous data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, ip_add, resume_score, convert(Predicted_Field using utf8), convert(User_level using utf8), city, state, country from user_data''')
                datanalys = cursor.fetchall()
                plot_data = pd.DataFrame(datanalys, columns=['Idt', 'IP_add', 'resume_score', 'Predicted_Field', 'User_Level', 'City', 'State', 'Country',])
                
                ### Total Users Count with a Welcome Message
                values = plot_data.Idt.count()
                st.success("관리자님 환영합니다! 총 %d " % values + "명의 사용자가 우리 도구를 사용했습니다 : )")                
                
                ### Fetch user data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, sec_token, ip_add, act_name, act_mail, act_mob, convert(Predicted_Field using utf8), Timestamp, Name, Email_ID, resume_score, Page_no, pdf_name, convert(User_level using utf8), convert(Actual_skills using utf8), convert(Recommended_skills using utf8), convert(Recommended_courses using utf8), city, state, country, latlong, os_name_ver, host_name, dev_user, toeic, github_address, blog, club, certificate from user_data''')
                data = cursor.fetchall()                

                st.header("**사용자 데이터**")
                df = pd.DataFrame(data, columns=['ID', 'Token', 'IP 주소', '이름', '메일', '전화번호', '예측된 분야', '타임스탬프',
                                             '예측된 이름', '예측된 메일', '이력서 점수', '총 페이지',  '파일 이름',   
                                             '사용자 레벨', '실제 기술', '권장 기술', '권장 코스',
                                             '도시', '행정 구역(도)', '국가', '위도 경도', '서버 OS', '서버 이름', '서버 사용자', '토익', '깃허브', '블로그', '동아리', '자격증'])
                
                ### Viewing the dataframe
                st.dataframe(df)
                
                ### Downloading Report of user_data in csv file
                st.markdown(get_csv_download_link(df,'User_Data.csv','보고서 다운로드'), unsafe_allow_html=True)

                ### Fetch feedback data from user_feedback(table) and convert it into dataframe
                cursor.execute('''SELECT * from user_feedback''')
                data = cursor.fetchall()

                st.header("**사용자 피드백 데이터**")
                df = pd.DataFrame(data, columns=['ID', '이름', '이메일', '피드백 점수', '코멘트', '타임스탬프'])
                st.dataframe(df)

                ### query to fetch data from user_feedback(table)
                query = 'select * from user_feedback'
                plotfeed_data = pd.read_sql(query, connection)                        

                ### Analyzing All the Data's in pie charts

                # fetching feed_score from the query and getting the unique values and total value count 
                labels = plotfeed_data.feed_score.unique()
                values = plotfeed_data.feed_score.value_counts()
                
                # Pie chart for user ratings
                st.subheader("**사용자 평점**")
                fig = px.pie(values=values, names=labels, title="1에서 5까지의 사용자 평점 차트 🤗", color_discrete_sequence=px.colors.sequential.Aggrnyl)
                st.plotly_chart(fig)

                # fetching Predicted_Field from the query and getting the unique values and total value count                 
                labels = plot_data.Predicted_Field.unique()
                values = plot_data.Predicted_Field.value_counts()

                # Pie chart for predicted field recommendations
                st.subheader("**예측 분야 추천을 위한 차트**")
                fig = px.pie(df, values=values, names=labels, title='기술에 따른 예측 분야의 파이 차트 👽', color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
                st.plotly_chart(fig)

                # fetching User_Level from the query and getting the unique values and total value count                 
                labels = plot_data.User_Level.unique()
                values = plot_data.User_Level.value_counts()

                # Pie chart for User's👨‍💻 Experienced Level
                st.subheader("**사용자의 경험 수준을 위한 차트**")
                fig = px.pie(df, values=values, names=labels, title="차트 📈 for 사용자의 👨‍💻 경험 수준", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig)

                # fetching resume_score from the query and getting the unique values and total value count                 
                labels = plot_data.resume_score.unique()                
                values = plot_data.resume_score.value_counts()

                # Pie chart for Resume Score
                st.subheader("**이력서 점수를 나타내는 차트**")
                fig = px.pie(df, values=values, names=labels, title='1부터 100까지 💯', color_discrete_sequence=px.colors.sequential.Agsunset)
                st.plotly_chart(fig)

                # fetching IP_add from the query and getting the unique values and total value count 
                labels = plot_data.IP_add.unique()
                values = plot_data.IP_add.value_counts()

                # Pie chart for Users
                st.subheader("**사용자 앱 사용 횟수를 나타내는 차트**")
                fig = px.pie(df, values=values, names=labels, title='IP 주소 기반 사용량 👥', color_discrete_sequence=px.colors.sequential.matter_r)
                st.plotly_chart(fig)

                # fetching City from the query and getting the unique values and total value count 
                labels = plot_data.City.unique()
                values = plot_data.City.value_counts()

                # Pie chart for City
                st.subheader("**도시를 나타내는 차트**")
                fig = px.pie(df, values=values, names=labels, title='도시 기반 사용량 🌆', color_discrete_sequence=px.colors.sequential.Jet)
                st.plotly_chart(fig)

                # fetching State from the query and getting the unique values and total value count 
                labels = plot_data.State.unique()
                values = plot_data.State.value_counts()

                # Pie chart for State 도로 번역했습니다
                st.subheader("**행정 구역(도)을 나타내는 차트**")
                fig = px.pie(df, values=values, names=labels, title='행정구역(도) 기반 사용량 🚉', color_discrete_sequence=px.colors.sequential.PuBu_r)
                st.plotly_chart(fig)

                # fetching Country from the query and getting the unique values and total value count 
                labels = plot_data.Country.unique()
                values = plot_data.Country.value_counts()

                # Pie chart for Country
                st.subheader("**국가를 나타내는 차트**")
                fig = px.pie(df, values=values, names=labels, title='국가 기반 사용량  🌏', color_discrete_sequence=px.colors.sequential.Purpor_r)
                st.plotly_chart(fig)

            ## For Wrong Credentials
            else:
                st.error("잘못된 ID 및 비밀번호가 제공되었습니다. ")



# Calling the main (run()) function to make the whole process run
run()
