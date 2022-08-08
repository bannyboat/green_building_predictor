from cProfile import label
import streamlit as st
import style
import numpy as np
import pandas as pd
import framework
from framework import *
from PIL import Image


st.markdown(
    f"""
<style>
    .reportview-container .main .block-container .css-10trblm e16nr0p33{{
        max-width: {1000}px;
        padding-top: {5}rem;
        padding-right: {0}rem;
        padding-left: {0}rem;
        padding-bottom: {0}rem;
    }}
    .reportview-container .main {{     

    }}   
    [data-testid="stImage"] img {{
        margin: 0 auto;
        max-width: 500px;
    }}
</style>
""",
    unsafe_allow_html=True,
)


style.display_app_header(main_txt='Green Building Project Cost & Duration Predictor 🔭',
                         sub_txt='An Artificial Intelligence based predictor for green building project cost and duration',is_sidebar=False)

# image = Image.open('img.jpg')
# new_image = image.resize((600, 600))
# st.image(new_image,width=1000)
# st.image("img.jpg")

style.display_app_header(main_txt='Building Types',
                         sub_txt="""
                         <p> 1. Commercial: Commerce and industry, office buildings </p>
                         <p> 2. Education: Schools and universities </p>
                         <p> 3. Government: Public safety facilities (e.g fire stations, ambulance facilities), law and order facilities (e.g court buildings) </p>
                         <p> 4. Mixed: Buildings that combine several funtions, such as residential and commercial, at once.
                         <p> 5. Residential; Hostels, hotels and quarters.
                         <p> 6. Other; Buildings that do not fall for any other category. </p>
                         """, is_sidebar=False)
style.display_app_header(main_txt='<p></p> Cost and Duration Prediction in Green Building Projects using Artificial Intelligence </p>' ,
                         sub_txt="""
                         """, is_sidebar=True)
#User Input Collection 
with st.sidebar:
    problem_selected = st.radio("Select what to Predict",("Final Cost","Actual Duration"))
    problemDic = {"Final Cost":"FC","Actual Duration":"AD"}
    problem = problemDic[problem_selected]
    st.header("Project Features")
    PT_selected = st.selectbox("Select Project Type", 
    ("Commercial","Educational","Government","Mixed-use","Residential","Other")) 
    PTDic = {"Residential":5,"Commercial":1,"Educational":2,"Government":3,"Mixed-use":4,"Other":6} 
    PT = PTDic[PT_selected]
    AS = st.number_input("Enter Area Size in Square Meters",value=8000.0,min_value = 1.0)
    AT_selected = st.selectbox("Select Area Type",("Floor Area","Site Area"))
    ATDic = {"Floor Area":0,"Site Area":1}
    AT = ATDic[AT_selected]
    OB = st.number_input("Enter Original Budget in Million HKD",value=135.9425,min_value = 0.1,step= 0.0001)
    PD = st.number_input("Enter Plan Duration in years",value=1.74,min_value = 0.1)
    SM_selected = st.selectbox("Select Start Month",("January", "February","March","April","May","June","July","August","September","October","November","December"))
    SMDic = {"January":1, "February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    SM = SMDic[SM_selected]
    SY = st.number_input("Enter Start Year",value=2012)
    SS = st.number_input("Enter Sustainable Site",value=76.0, min_value=50.0, max_value = 100.0)
    MW = st.number_input("Enter Materials and Waste",value=50.0, min_value=9.0,max_value = 100.0)
    EU = st.number_input("Enter Energy Use",value=68.0, min_value=50.0, max_value = 100.0)
    WU = st.number_input("Enter Water Use",value=71.0, min_value=30.0, max_value = 100.0)
    HWB = st.number_input("Enter Health and Wellbeing",value=82.0, min_value=50.0, max_value = 100.0)
    IA = st.number_input("Enter Innovations and Additions",value=5.0, min_value=1.0, max_value = 6.0)


#Display project type logo 



# Number of cold, hot and rainy days is calculated according to HK Observatory data
# So first we load this data. Note that we only have info up to the Dec 2020. 
CD = framework.calculate_days_of_weather(SM, SY, PD, "cold")  
HD = framework.calculate_days_of_weather(SM, SY, PD, "hot")  
RD = framework.calculate_days_of_weather(SM, SY, PD, "rain") 
# Combining features into one input X
X = [PT, AS, AT, OB, PD, SM, SY, SS, MW, EU, WU, HWB, IA, CD, HD, RD]
X_display = [PT, AS, AT, OB, PD, SM, SY, SS, MW, EU, WU, HWB, IA, CD[0], HD[0], RD[0]]
columns = ["PT", "AS", "AT", "OB", "PD", "SM", "SY", "SS", "MW", "EU", "WU", "HWB", "IA", "CD", "HD", "RD"]
df = pd.DataFrame([X_display],columns=columns)
#Display Dataframe 
st.markdown("### **Features and inputed values table**")
st.dataframe(df)

# J = [1,200000.00,1.00,569.90,1.10,3,1999,65.00,23.00,83.00,50.00,92.00,5.00,23,6,9]
# print("prediction",framework.predict(J, problem))


# Predicting the target according to the problem, which is given in line 26 

prediction = framework.predict(X, problem)
if(problem=="FC"):
    st.markdown("### Predicted Final Cost: HK$ "+str(np.round(prediction[0],10)[0]) + "M")
else:
    st.markdown("### Predicted project Duration: "+str(np.round(prediction[0],2)[0]) + " Yrs")


# st.metric(label="Predicted " + problem_selected + " in HNK",value=np.round(prediction[0],2))







    # PT = st.slider('', 0, 130, 25) 