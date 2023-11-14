# 모든 패키지 import를 모아 놓은 py


import nltk
nltk.download('stopwords')
###### Packages Used ######
import streamlit as st # core package used in this project
import pandas as pd
import base64, random
import time,datetime
import pymysql
import os
import socket
import platform
import geocoder
import secrets
import io,random
import plotly.express as px # to create visualisations at the admin session
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
# libraries used to parse the pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos

# 초기 이후 생겨난 py file
# 만든 함수는 아래와 같이 다 명시해줘야 다른 파일에서 인식합니다.
# * 사용하면 안됩니다(이유는 모르겠음)
from Functions import get_csv_download_link,show_pdf,pdf_reader,course_recommender,insert_data,insertf_data 
