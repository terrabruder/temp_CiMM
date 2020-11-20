############################################################
#Cirrhosis Mortality Model (CiMM) Scoring Application
############################################################

#This application predicts risk of mortality as a result of
#complications with Cirrhosis. The model relies on a
#discrete-time-to-event logistic regression. For questions
#or clarifications on code, please consult 
#with thomjtaylor@gmail.com
#
#The application relies primarily on Bokeh built in Python 3.7.6
#and the the essential libraries necessary to run the application
#in any environment will be
#1. Numpy
#2. Pandas
#3. Scipy
#4. Bokeh

############################################################
#import core functiones
############################################################

#%reset

#from os.path import dirname, join
from scipy.stats import norm
from scipy.special import expit

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row, WidgetBox, layout
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, CheckboxGroup, HoverTool, Panel, CustomJS, DataTable, NumberFormatter, RangeSlider, TableColumn, LinearColorMapper, PrintfTickFormatter, BasicTicker, ColorBar, NumeralTickFormatter
from bokeh.models.widgets import MultiSelect, Tabs
from bokeh.plotting import output_file, figure, show
from bokeh.models import LinearAxis, Range1d, FactorRange, ranges, Button
#from bokeh.palettes import RdYlGn, RdYlBu, RdBu, Spectral6
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.transform import transform, linear_cmap
from bokeh.models.scales import CategoricalScale

#############################################################
##file location references
#############################################################
#bdat = pd.read_csv(join(dirname(__file__), "mortality_scoring_B_coefficients.csv"))
#vcovdat = pd.read_csv(join(dirname(__file__), "mortality_scoring_coefficients_vcov.csv"))
bdat = pd.read_csv("mortality_scoring_B_coefficients.csv")
vcovdat = pd.read_csv("mortality_scoring_coefficients_vcov.csv")

print(len(bdat))
print(vcovdat.shape)

Bnames = bdat.iloc[:,1]
Bcoefs = bdat.iloc[:,2]

############################################################
#multiway checkbox of the metric variable
############################################################

circom_labels = [
    "History of Chronic Obstructive Pulmonary Disease (COPD)",	
    "History of Acute Myocardial Infarction (AMI)",	
    "Acute Myocardial Infarction (AMI)",	
    "History of Peripheral Arterial Disease (PAD)",	
    "History of Epilepsy",	
    "History of Substance Abuse other than Alcohol (SUD)",	
    "History of Heart Failure",	
    "History of Non-metastatic or Hematologic Cancer",	
    "Active Non-metastatic or Hematologic Cancer",	
    "Inactive Metastatic Cancer",	
    "Active Metastatic Cancer",	
    "History of Chronic Kidney Disease (CKD)"]

#############################################################
#############################################################
#############################################################
#Widget settings for input values
#############################################################
#############################################################
#############################################################

visits_title="Has the patient had more 3 or more outpatient care visits in the past year?"
age_title="Age (Years)"
black_title="Does the patient identify as Black (African American)?"
sodium_title="Sodium mEq/L"
bilirubin_title="Total Bilirubin mg/dL"
platelets_title="Platelets per nL"
albumin_title="Albumin g/dL"
hemoglobin_title="Hemoglobin g/dL"
astalt_title="Does the Patient have an AST/ALT Ratio greater than 2.0?"
he_title="Does the Patient have Hepatic Encephalopathy?"
ascites_title="Has the patient developed Ascites?"
hcc_title="Has the patient developed Hepato-Cellular Carcinoma (HCC)?"

#Circom elements selection ############################
circom_selection = CheckboxGroup(labels=circom_labels, active = []) #active = [0,1,2,3,4,5,6,7,8,9,10,11])

#3 or more outpatient care visits ############################
select_visits = Select(title=visits_title, 
                      value="No", 
                      options=["No", "Yes"])
#age #########################################################
age_slider = Slider(start=18, end=100, value=55, step=1, title=age_title)
#race (Black) ################################################
select_black = Select(title=black_title, 
                      value="No", 
                      options=["No", "Yes"])
##############################################################
##labs
##############################################################
##### Sodium #####
##### Bilirubin #####
##### Platelets (in 50 unit intervals) #####
##### Albumin #####
##### Hemoglobin #####
sodium_slider = Slider(start=110, end=160, value=140, step=1, title=sodium_title)
bili_slider = Slider(start=.1, end=20, value=3, step=.1, title=bilirubin_title)
plat_slider = Slider(start=1, end=500, value=100, step=10, title=platelets_title) #150 to 400 × 10^-9/L
alb_slider = Slider(start=.1, end=10, value=4.5, step=.1, title=albumin_title) #3.4 to 5.4 g/dL is normal
hemo_slider = Slider(start=.1, end=20, value=10.0, step=.1, title=hemoglobin_title)
##### AST/ALT Ratio >2  ######################################
select_astalt = Select(title=astalt_title, 
                      value="No", 
                      options=["No", "Yes"])
##### History of Hepatic Encephalopathy ######################
select_he = Select(title=he_title, 
                      value="No", 
                      options=["No", "Yes"])
##### History of Ascites #####################################
select_ascites = Select(title=ascites_title, 
                      value="No", 
                      options=["No", "Yes"])
##### HCC ####################################################
select_hcc = Select(title=hcc_title, 
                      value="No", 
                      options=["No", "Yes"])
#Confidence Interval setting #################################
confint_slider = Slider(start=90, end=99, value=95, step=1, title="Select Confidence Interval (e.g., 95% CI)")

#############################################################
#############################################################
#############################################################
##create prediction dataset
#############################################################
#############################################################
#############################################################

##### COLUMN ORDERING FOR SCORING ###########################
##### outpatient visits in past year 3 or more #####
##### age at index date #####
##### Black race #####
##### Sodium #####
##### Bilirubin #####
##### Platelets (in 50 unit intervals) #####
##### Albumin #####
##### Hemoglobin #####
##### AST/ALT Ratio >2 #####
##### Circom factor 1 #####
##### Circom factor 2 #####
##### Circom factor 3 #####
##### Circom factor 4 #####
##### Circom factor 5 #####
##### Circom factor 6 #####
##### History of Hepatic Encephalopathy #####
##### History of Ascites #####
##### HCC #####

def make_dataset(circom_selection,
                 select_visits,
                 age_slider,
                 select_black,
                 sodium_slider,
                 bili_slider,
                 plat_slider,
                 alb_slider,
                 hemo_slider,
                 select_astalt,
                 select_he,
                 select_ascites,
                 select_hcc,
                 confint_slider,
                 visits_title,
                 age_title,
                 black_title,
                 sodium_title,
                 bilirubin_title,
                 platelets_title,
                 albumin_title,
                 hemoglobin_title,
                 astalt_title,
                 he_title,
                 ascites_title,
                 hcc_title):
    #############################################################
    ##compute Circom score
    #############################################################
    circom_selected_orig = np.array(circom_selection.active)
    len(circom_selected_orig)
    
    if len(circom_selected_orig)==0: 
        circom_selected=np.array([-1])
    else: 
        circom_selected=circom_selected_orig
    
    circom_selected_orig
    circom_selected
    
    hx_copd = max(np.where(circom_selected==0, 1, 0))
    hx_ami = max(np.where(circom_selected==1, 1, 0))
    ami = max(np.where(circom_selected==2, 1, 0))
    hx_pad = max(np.where(circom_selected==3, 1, 0))
    hx_epi = max(np.where(circom_selected==4, 1, 0))
    hx_sud = max(np.where(circom_selected==5, 1, 0))
    hx_hf = max(np.where(circom_selected==6, 1, 0))
    hx_nonmet_hemat_cancer = max(np.where(circom_selected==7, 1, 0))
    active_nonmet_hemat_cancer = max(np.where(circom_selected==8, 1, 0))
    hx_inactive_met = max(np.where(circom_selected==9, 1, 0))
    active_met = max(np.where(circom_selected==10, 1, 0))
    hx_ckd = max(np.where(circom_selected==11, 1, 0))
    
    sum_all_comorbidites = hx_copd + hx_ami + ami + hx_pad + hx_epi + hx_sud + hx_hf + hx_nonmet_hemat_cancer + active_nonmet_hemat_cancer + hx_inactive_met + active_met + hx_ckd
    #need to exclude inactive metastatic cancer and CKD since part of the "3 + 0" and "3 + 1" scores
    #hx_inactive_met, hx_ckd
    sum_hx_comorbidities = hx_copd + hx_ami + hx_pad + hx_epi + hx_sud + hx_hf + hx_nonmet_hemat_cancer 
    
    sum_all_comorbidites
    sum_hx_comorbidities
    
    #CirCom Algorithm (https://www.gastrojournal.org/article/S0016-5085(13)01347-4/pdf)
    #1. Does the patient have COPD, AMI, PAD, Epilepsy, SUD Heart Failur, Cancer, or CKD?
    #    No = "0"
    #    Yes = Step 2
    #2. Does the patient have Active Metastatic Cancer?
    #    No = Step 3
    #    Yes = Step 2a
    #2a. Does the patient have MORE THAN 1 of the listed comorbidities?
    #    No = "5 + 0"
    #    Yes = "5 + 1"
    #3. Does the patient have Active Mycardial Infarction, Active Non-Metatastatic or Hematologic Cancer, Inactive Metatstatic Cancer, or CKD?
    #    No = Step 4
    #    Yes = Step 3a
    #3a. Does the patient have MORE THAN one of the listed comorbidities?
    #    No = "3 + 0"
    #    Yes = "3 + 1"
    #4. Does the patient have MORE THAN one of the listed comorbidities?
    #    No = "1 + 0"
    #    Yes = "1 + 1"
    
    if sum_all_comorbidites==0: 
        circom_score = "0"
    elif ((active_met==1) & (sum_hx_comorbidities<=1)):
        circom_score = "5 + 0"
    elif ((active_met==1) & (sum_hx_comorbidities>1)): 
        circom_score = "5 + 1"
    elif ((active_met==0) & ((ami==1) | (active_nonmet_hemat_cancer==1) | (hx_inactive_met==1) | (hx_ckd==1)) & (sum_hx_comorbidities<=1)):
        circom_score = "3 + 0"
    elif ((active_met==0) & ((ami==1) | (active_nonmet_hemat_cancer==1) | (hx_inactive_met==1) | (hx_ckd==1)) & (sum_hx_comorbidities>1)):
        circom_score = "3 + 1"
    elif ((active_met==0) & ((ami==0) & (active_nonmet_hemat_cancer==0) & (hx_inactive_met==0) & (hx_ckd==0)) & (sum_hx_comorbidities<=1)):
        circom_score = "1 + 0"
    elif ((active_met==0) & ((ami==0) & (active_nonmet_hemat_cancer==0) & (hx_inactive_met==0) & (hx_ckd==0)) & (sum_hx_comorbidities>1)):
        circom_score = "1 + 1"
    else: circom_score = "0"
    
    circom_score
    
    if circom_score=="0":
        circom_num = 0
    elif circom_score=="5 + 0":
        circom_num = 5
    elif circom_score=="5 + 1":
        circom_num = 6
    elif circom_score=="3 + 0":
        circom_num = 3
    elif circom_score=="3 + 1":
        circom_num = 4
    elif circom_score=="1 + 0":
        circom_num = 1
    elif circom_score=="1 + 1":
        circom_num = 2
    else: circom_num = 0
    
    circom_num
    
    ##### CirCom score inputs to scoring equation ################
    ##### Circom factor 1 #####
    circom1 = np.where(circom_num==1, 1, 0)
    ##### Circom factor 2 #####
    circom2 = np.where(circom_num==2, 1, 0)
    ##### Circom factor 3 #####
    circom3 = np.where(circom_num==3, 1, 0)
    ##### Circom factor 4 #####
    circom4 = np.where(circom_num==4, 1, 0)
    ##### Circom factor 5 #####
    circom5 = np.where(circom_num==5, 1, 0)
    ##### Circom factor 6 #####
    circom6 = np.where(circom_num==6, 1, 0)   
    #############################################################
    ##create values from other selected elements of the model trained scoring system
    #############################################################      
    visitval = np.where(select_visits.value=="Yes", 1, 0)
    ageval = age_slider.value
    raceval = np.where(select_black.value=="Yes", 1, 0)
    sodiumval = sodium_slider.value
    bilirubinval = bili_slider.value
    plateletsval = plat_slider.value/50 #since training occurreed with all platelet values/50 for effect size interpretation manipulation
    albuminval = alb_slider.value
    hemoglobinval = hemo_slider.value
    astaltval = np.where(select_astalt.value=="Yes", 1, 0)
    heval = np.where(select_he.value=="Yes", 1, 0)
    ascitesval = np.where(select_ascites.value=="Yes", 1, 0)
    hccval = np.where(select_hcc.value=="Yes", 1, 0)
    
    risk_profile_text = str("Risk Profile: CirCom Score=" + str(circom_score) + ", " + 
        visits_title + "=" + str(select_visits.value) + ", " + 
        age_title  + "=" + str(age_slider.value) + ", " + 
        black_title + "=" + str(select_black.value) + ", " + 
        sodium_title + "=" + str(sodium_slider.value) + ", " + 
        bilirubin_title + "=" + str(bili_slider.value) + ", " + 
        platelets_title + "=" + str(plat_slider.value) + ", " + 
        albumin_title + "=" + str(alb_slider.value) + ", " + 
        hemoglobin_title + "=" + str(hemo_slider.value) + ", " + 
        astalt_title + "=" + str(select_astalt.value) + ", " + 
        he_title + "=" + str(select_he.value) + ", " + 
        ascites_title + "=" + str(select_ascites.value) + ", " + 
        hcc_title + "=" + str(select_hcc.value))
    #############################################################
    ##create scoring dataset
    #############################################################   
    valseries = pd.Series([visitval, ageval, raceval, 
                           sodiumval, bilirubinval, plateletsval, albuminval, hemoglobinval, astaltval, 
                           circom1, circom2, circom3, circom4, circom5, circom6, 
                           heval, ascitesval, hccval])
    
    y1vec = pd.concat([pd.Series([1, 0, 0, 0, 0, 0, 0, 0]), valseries], axis=0)
    y2vec = pd.concat([pd.Series([1, 1, 0, 0, 0, 0, 0, 0]), valseries], axis=0)
    y3vec = pd.concat([pd.Series([1, 1, 1, 0, 0, 0, 0, 0]), valseries], axis=0)
    y4vec = pd.concat([pd.Series([1, 1, 1, 1, 0, 0, 0, 0]), valseries], axis=0)
    y5vec = pd.concat([pd.Series([1, 1, 1, 1, 1, 0, 0, 0]), valseries], axis=0)
    y6vec = pd.concat([pd.Series([1, 1, 1, 1, 1, 1, 0, 0]), valseries], axis=0)
    y7vec = pd.concat([pd.Series([1, 1, 1, 1, 1, 1, 1, 0]), valseries], axis=0)
    y8vec = pd.concat([pd.Series([1, 1, 1, 1, 1, 1, 1, 1]), valseries], axis=0)
    
    obsdat = pd.concat([y1vec, y2vec, y3vec, y4vec, y5vec, y6vec, y7vec, y8vec], axis=1).T
    
    #Confidence Interval selected setting #################################
    confintval = confint_slider.value
    two_sided_alpha = (100-confintval)*.01
    
    se_logit_est = []
    for i in range(len(obsdat)):
        tempvals = obsdat.iloc[i,]
        tempSE = np.sqrt(np.dot(np.dot(tempvals.T, vcovdat), tempvals))
        tempSEdat = pd.DataFrame({"var_index": vcovdat.index[i], "se": tempSE}, index=[i])
        se_logit_est.append(tempSEdat)
    std_errors_dat = pd.concat(se_logit_est, axis=0).reset_index(drop=True)
    lower_ppf = two_sided_alpha/2
    upper_ppf = 1-(two_sided_alpha/2)
    
    eta = pd.Series(np.dot(obsdat, Bcoefs)).astype(float)
    riskprob = pd.Series(1/(1+np.exp(-eta)))
    
    logit_pred = pd.DataFrame(eta).reset_index(drop=True)
    proba_pred = pd.DataFrame(riskprob).reset_index(drop=True)
    #quantile_var = pd.DataFrame(std_errors_dat.loc[:,["var_index"]]).reset_index(drop=True)
    selower = pd.DataFrame(std_errors_dat.loc[:,["se"]] * norm.ppf(lower_ppf)).reset_index(drop=True)
    seupper = pd.DataFrame(std_errors_dat.loc[:,["se"]] * norm.ppf(upper_ppf)).reset_index(drop=True)
    logit_pred_ll = pd.DataFrame(np.array(logit_pred) + np.array(selower)).reset_index(drop=True)
    logit_pred_ul = pd.DataFrame(np.array(logit_pred) + np.array(seupper)).reset_index(drop=True)
    proba_pred_ll = pd.DataFrame(expit(logit_pred_ll)).reset_index(drop=True).reset_index(drop=True)
    proba_pred_ul = pd.DataFrame(expit(logit_pred_ul)).reset_index(drop=True)
    
    pdat = pd.concat([logit_pred, logit_pred_ll, logit_pred_ul, proba_pred, proba_pred_ll, proba_pred_ul], axis=1).round(4)
    pdat.columns = ["logit", "logit_ll", "logit_ul", "probability", "probability_ll", "probability_ul"]
    pdat["percent"] = np.round(pdat["probability"]*100, 1)
    pdat["percent_ll"] = np.round(pdat["probability_ll"]*100, 1)
    pdat["percent_ul"] = np.round(pdat["probability_ul"]*100, 1)
    pdat["year"] = pd.Series(pdat.index+1).astype(str)
    pdat["patyear"] = pd.Series(np.repeat("The Patient's Risk of Death in ", len(pdat))) + pdat["year"].astype(str) + pd.Series([" Year", " Years", " Years", " Years", " Years", " Years", " Years", " Years"])
    pdat["Estimate"] = pdat["percent"].astype(str) + pd.Series(np.repeat("% (", len(pdat))) + pdat["percent_ll"].astype(str) + pd.Series(np.repeat("% - ", len(pdat))) + pdat["percent_ul"].astype(str) + pd.Series(np.repeat("%)", len(pdat)))
    pdat["Interpretation"] = pdat["patyear"] +  pd.Series(np.repeat(" is ", len(pdat))) + pdat["Estimate"]    
#    pdat["circom_value"] = pd.Series(np.repeat("CirCom Score = ", len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(circom_score), len(pdat))) 
#    pdat["visits_value"] = pd.Series(np.repeat(visits_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(select_visits.value), len(pdat)))
#    pdat["age_value"] = pd.Series(np.repeat(age_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(age_slider.value), len(pdat)))
#    pdat["black_value"] = pd.Series(np.repeat(black_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(select_black.value), len(pdat)))
#    pdat["sodium_value"] = pd.Series(np.repeat(sodium_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(sodium_slider.value), len(pdat)))
#    pdat["bilirubin_value"] = pd.Series(np.repeat(bilirubin_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(bili_slider.value), len(pdat)))
#    pdat["platelets_value"] = pd.Series(np.repeat(platelets_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(plat_slider.value), len(pdat)))
#    pdat["albumin_value"] = pd.Series(np.repeat(albumin_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(alb_slider.value), len(pdat)))
#    pdat["hemoglobin_value"] = pd.Series(np.repeat(hemoglobin_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(hemo_slider.value), len(pdat)))
#    pdat["astalt_value"] = pd.Series(np.repeat(astalt_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(select_astalt.value), len(pdat)))
#    pdat["he_value"] = pd.Series(np.repeat(he_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(select_he.value), len(pdat)))
#    pdat["ascites_value"] = pd.Series(np.repeat(ascites_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(select_ascites.value), len(pdat)))
#    pdat["hcc_value"] = pd.Series(np.repeat(hcc_title, len(pdat))) + pd.Series(np.repeat(" ", len(pdat))) + pd.Series(np.repeat(str(select_hcc.value), len(pdat)))
    pdat["circom_value"] = pd.Series(np.repeat(str(circom_score), len(pdat))) 
    pdat["visits_value"] = pd.Series(np.repeat(str(select_visits.value), len(pdat)))
    pdat["age_value"] = pd.Series(np.repeat(str(age_slider.value), len(pdat)))
    pdat["black_value"] = pd.Series(np.repeat(str(select_black.value), len(pdat)))
    pdat["sodium_value"] = pd.Series(np.repeat(str(sodium_slider.value), len(pdat)))
    pdat["bilirubin_value"] = pd.Series(np.repeat(str(bili_slider.value), len(pdat)))
    pdat["platelets_value"] = pd.Series(np.repeat(str(plat_slider.value), len(pdat)))
    pdat["albumin_value"] = pd.Series(np.repeat(str(alb_slider.value), len(pdat)))
    pdat["hemoglobin_value"] = pd.Series(np.repeat(str(hemo_slider.value), len(pdat)))
    pdat["astalt_value"] = pd.Series(np.repeat(str(select_astalt.value), len(pdat)))
    pdat["he_value"] = pd.Series(np.repeat(str(select_he.value), len(pdat)))
    pdat["ascites_value"] = pd.Series(np.repeat(str(select_ascites.value), len(pdat)))
    pdat["hcc_value"] = pd.Series(np.repeat(str(select_hcc.value), len(pdat)))
    pdat["Risk_Profile"] = pd.Series(np.repeat(str(risk_profile_text), len(pdat)))
    return pdat

#temp = make_dataset(circom_selection,
#                 select_visits,
#                 age_slider,
#                 select_black,
#                 sodium_slider,
#                 bili_slider,
#                 plat_slider,
#                 alb_slider,
#                 hemo_slider,
#                 select_astalt,
#                 select_he,
#                 select_ascites,
#                 select_hcc,
#                 confint_slider,
#                 visits_title,
#                 age_title,
#                 black_title,
#                 sodium_title,
#                 bilirubin_title,
#                 platelets_title,
#                 albumin_title,
#                 hemoglobin_title,
#                 astalt_title,
#                 he_title,
#                 ascites_title,
#                 hcc_title)

#############################################################
#############################################################
#############################################################
##create an initial source shell dictionary
#############################################################
#############################################################
#############################################################

source = ColumnDataSource(data=dict(
        logit=[],
        logit_ll=[],
        logit_ul=[],
        probability=[],
        probability_ll=[],
        probability_ul=[],
        percent=[],
        percent_ll=[],
        percent_ul=[],
        year=[],
        patyear=[],
        Estimate=[],
        Interpretation=[],
        circom_value=[],
        visits_value=[],
        age_value=[],
        black_value=[],
        sodium_value=[],
        bilirubin_value=[],
        platelets_value=[],
        albumin_value=[],
        hemoglobin_value=[],
        astalt_value=[],
        he_value=[],
        ascites_value=[],
        hcc_value=[],
        Risk_Profile=[]
        )
)

#############################################################
#############################################################
#############################################################
##build reactive table
#############################################################
#############################################################
#############################################################

#############################################################
##create an inline datatable structure
#############################################################
columns = [
        #TableColumn(field="year", title="Year"),
        TableColumn(field="Interpretation", title="Risk Estimate"),
        TableColumn(field="Risk_Profile", title="Risk Profile")
        ]
#############################################################
##create column data source to reference for output figures and tables
##best practice with Bokeh is to make a dictionary datasource
#############################################################
table = DataTable(source=source,
                  columns=columns,
                  #reorderable=False,
                  width=600,
                  height=600
                  #sizing_mode="stretch_width"
                  )
#############################################################
##updater function when new checkboxes are selected
#############################################################

def update():
    #updated_metrics = [metrics_selection.labels[i] for i in metrics_selection.active]
    #df = make_dataset(DATAFRAME=simdat, SELECTED_METRICS_LIST=updated_metrics)
    df = make_dataset(circom_selection,
                 select_visits,
                 age_slider,
                 select_black,
                 sodium_slider,
                 bili_slider,
                 plat_slider,
                 alb_slider,
                 hemo_slider,
                 select_astalt,
                 select_he,
                 select_ascites,
                 select_hcc,
                 confint_slider,
                 visits_title,
                 age_title,
                 black_title,
                 sodium_title,
                 bilirubin_title,
                 platelets_title,
                 albumin_title,
                 hemoglobin_title,
                 astalt_title,
                 he_title,
                 ascites_title,
                 hcc_title)
    source.data = dict(
            logit=df["logit"],
            logit_ll=df["logit_ll"],
            logit_ul=df["logit_ul"],
            probability=df["probability"],
            probability_ll=df["probability_ul"],
            probability_ul=df["probability_ul"],
            percent=df["percent"],
            percent_ll=df["percent_ll"],
            percent_ul=df["percent_ul"],
            year=df["year"],
            patyear=df["patyear"],
            Estimate=df["Estimate"],
            Interpretation=df["Interpretation"],
            circom_value=df["circom_value"],
            visits_value=df["visits_value"],
            age_value=df["age_value"],
            black_value=df["black_value"],
            sodium_value=df["sodium_value"],
            bilirubin_value=df["bilirubin_value"],
            platelets_value=df["platelets_value"],
            albumin_value=df["albumin_value"],
            hemoglobin_value=df["hemoglobin_value"],
            astalt_value=df["astalt_value"],
            he_value=df["he_value"],
            ascites_value=df["ascites_value"],
            hcc_value=df["hcc_value"],
            Risk_Profile=df["Risk_Profile"]
        )


#############################################################
#############################################################
#############################################################
##build reactive glyph
#############################################################
#############################################################
#############################################################

#############################################################
##create hover tooltips describing each point in the plot
#############################################################

TOOLTIPS=[('Risk', '@Interpretation'),
          ('CirCom Score', '@circom_value'),
          ('Outpatient Care', '@visits_value'),
          ('Age', '@age_value'),
          ('Racial Identification', '@black_value'),
          ('Sodium', '@sodium_value'),
          ('Bilirubin', '@bilirubin_value'),
          ('Platelets', '@platelets_value'),
          ('Albumin', '@albumin_value'),
          ('Hemoglobin', '@hemoglobin_value'),
          ('AST/ALT Ratio', '@astalt_value'),
          ('Hepatic Encephalopathy', '@he_value'),
          ('Ascites', '@ascites_value'),
          ('HCC', '@hcc_value')]

#############################################################
##color palette for waffle plot figure
#############################################################

colors = ["#0D4D4D", "#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]

mapper = LinearColorMapper(palette=colors, 
                       low=0, 
                       high=1.00)

color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                 ticker=BasicTicker(desired_num_ticks=10), #len(colors)),
                 orientation='vertical',
                 formatter=NumeralTickFormatter(format='0%'),
                 label_standoff=10)

##############################################################
###initialize a figure
##############################################################          
#initial_metrics = [metrics_selection.labels[i] for i in metrics_selection.active]
initial_df = make_dataset(circom_selection,
                 select_visits,
                 age_slider,
                 select_black,
                 sodium_slider,
                 bili_slider,
                 plat_slider,
                 alb_slider,
                 hemo_slider,
                 select_astalt,
                 select_he,
                 select_ascites,
                 select_hcc,
                 confint_slider,
                 visits_title,
                 age_title,
                 black_title,
                 sodium_title,
                 bilirubin_title,
                 platelets_title,
                 albumin_title,
                 hemoglobin_title,
                 astalt_title,
                 he_title,
                 ascites_title,
                 hcc_title)

p = figure(sizing_mode="stretch_width", 
            width=600,
            height=400,
            title="Cirrhosis Mortality Model Risk Profiling",
            y_axis_label = 'Mortality Risk', 
            x_axis_label = "Years from Today's Date",
            y_range = Range1d(-0.01, 1.01),
            #x_range = FactorRange(factors=unique_hospital_list),
            x_range = FactorRange(factors=["1", "2", "3", "4", "5", "6", "7", "8"]),
            #x_scale = CategoricalScale(),
            #toolbar_location=None,
            tooltips=TOOLTIPS, 
            x_axis_location="below")
p.yaxis[0].ticker.desired_num_ticks = 10
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.xaxis.major_label_text_font_size = "10pt"
p.yaxis.major_label_text_font_size = "30pt"
p.axis.axis_label_text_font_size = "10pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = 1.0
p.yaxis.visible = False
p.ygrid.visible = True
p.add_layout(color_bar, 'left')

p.line(y="probability", 
         x="year", 
         source=source,
         line_color="firebrick", 
         line_width=1, 
         line_alpha=.8)
#glyph = Line(x="x", y="y", line_color="#f46d43", line_width=6, line_alpha=0.6)
#plot.add_glyph(source, glyph)

p.circle(y="probability", 
         x="year", 
         source=source, 
         size=15, 
         color=transform('probability', mapper), 
         #line_color=transform('probability', mapper)
         line_color="gray"
         )

#############################################################
#############################################################
#############################################################
##run update on updated controls toggled by end user
#############################################################
#############################################################
#############################################################

controls_active = [circom_selection,
                 select_visits,
                 age_slider,
                 select_black,
                 sodium_slider,
                 bili_slider,
                 plat_slider,
                 alb_slider,
                 hemo_slider,
                 select_astalt,
                 select_he,
                 select_ascites,
                 select_hcc,
                 confint_slider]
#### for automatic updating, uncomment this ####
#for control in controls_active: 
#    control.on_change('active', lambda attr, old, new: update_reference_dataset())
    
bt = Button(
    label="Click to Update\nRisk Prediction",
    button_type= "primary",
    width=75
)

bt.on_click(update)


#############################################################
#############################################################
#############################################################
##format layout of application presentation
#############################################################
#############################################################
#############################################################

############################################################
#reference html heading file
############################################################

#text=open(join(dirname(__file__), 'application_title_description.html')).read(), 
heading = Div(
        text=open('application_title_description.html').read(), 
        height=100, 
        sizing_mode="stretch_width"
        )

############################################################
#decription of elements with Div
############################################################

circom_description = Div(#text=open('application_title_description.html').read(),
        text = """CirCom Score Components <a href="https://www.gastrojournal.org/article/S0016-5085(13)01347-4/pdf">(Validation Reference)</a>:""",
        height=20, 
        sizing_mode="stretch_width")

checkbox_description = Div(text="<b>Select CirCom Data Elements:</b>", 
              height=20, sizing_mode="stretch_width")

table_description = Div(text="<b>Output Table of Unique Risk Profiles Based on Factors Selected:</b>", 
              height=20, sizing_mode="stretch_width")

############################################################
#layout elements
############################################################

colcontrols = column(bt,
                     checkbox_description,
                     circom_description,
                     circom_selection,
                     select_visits,
                     age_slider,
                     select_black,
                     sodium_slider,
                     bili_slider,
                     plat_slider,
                     alb_slider,
                     hemo_slider,
                     select_astalt,
                     select_he,
                     select_ascites,
                     select_hcc,
                     confint_slider, width=00)


coltable = column(table_description, table, sizing_mode="stretch_both")
colestimates = column(p,coltable, width=600, height=600, sizing_mode="stretch_both")

rowdisp = row(colcontrols, colestimates, sizing_mode="stretch_both")
lay = layout([[heading],
        [rowdisp]
        ]
)
      
#############################################################
##deployment code 
#############################################################

curdoc().add_root(lay)
curdoc().title = "Cirrhosis Mortality Model Risk Scoring"


#bokeh serve C:/Users/ketam/CMMscore/main.py --show
###server deployment
###https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html
#
#run in cmd
#navigate to directory CMMscore direcotory
#bokeh serve main.py --show
