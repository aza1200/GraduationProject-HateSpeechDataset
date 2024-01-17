import pandas as pd

def beep2df(df):
    beep_df = df

    beep_sex_discrimination_series = beep_df[
        (beep_df['contain_gender_bias'] ==  True) & (beep_df['hate'].isin(['hate', 'offensive']))
    ]['comments']
    common_speach_beep_seires = beep_df[beep_df['hate']=='none']['comments']


    sex_beep_df = pd.DataFrame(beep_sex_discrimination_series, columns=["comments"])
    common_beep_df = pd.DataFrame(common_speach_beep_seires, columns=["comments"])


    class_list_sex = [[1] for _ in range(len(sex_beep_df))]
    class_list_common = [[0] for _ in range(len(common_beep_df))]


    sex_beep_df["class"] = class_list_sex
    common_beep_df["class"] = class_list_common

    sex_beep_df.rename(columns={'comments': 'text'}, inplace=True)
    common_beep_df.rename(columns={'comments': 'text'}, inplace=True)

    labeled_beep_df = pd.concat([sex_beep_df, common_beep_df], ignore_index=True)
    labeled_beep_df['class'] = labeled_beep_df['class'].apply(lambda x: x[0])
    
    return labeled_beep_df

def hatescore2df(df):
    hate_score_df = df

    def lambda_function(x):
        x = eval(x)[0]
        if x == '남성' or x == '여성/가족':
            return "남성/여성"
        else:
            return x

    # hate_score 에서 class label 하나인것만 가져오기 
    hate_score_df = hate_score_df[hate_score_df['class'].apply(lambda x: len(eval(x)) == 1)]
    hate_score_df['class'] = hate_score_df['class'].apply(lambda x: lambda_function(x) )
    hate_score_df.reset_index(drop=True, inplace=True)

    return hate_score_df

# KMHAS_df
# 0: 출신
# 1: 외모
# 2: 정치성향
# 3 : 혐오욕설
# 4 : 연령 차별
# 5 : 성차별
# 6: 인종차별 
# 7 : 종교차별
# 8 : 일반 발언 -> 0
def kmhas2df(df):
    kmhas_df = df
    kmhas_df.rename(columns={'label': 'class'}, inplace=True)
    kmhas_df = kmhas_df[kmhas_df['class'].apply(lambda x: len(eval(x)) == 1)]
    kmhas_df.reset_index(drop=True,inplace=True)
    class_change_dict = {
    0: "출신" ,1:"외모" , 2:"정치성향", 3:"혐오욕설",
    4: "연령 차별", 5:"성차별" , 6:"인종차별", 7:"종교차별",
    8: "일반 발언"
    }

    def lambda_function(x):
        x = eval(x)[0]
        return class_change_dict[x]

    kmhas_df['class'] = kmhas_df['class'].apply(lambda x: lambda_function(x))
    
    return kmhas_df

def kold2df(df):
    kold_df = df

    class_list = []
    text_list = []

    for idx in range(len(kold_df)):
        comment = kold_df.loc[idx].comment
        grps = kold_df.loc[idx].GRP
        is_offensive = kold_df.loc[idx].OFF
        tmp_class = []
        will_append = False
        if is_offensive == False:
            tmp_class.append(0)
            will_append = True
        else:
            if pd.isna(kold_df.loc[idx, 'GRP']) == True:
                continue
            labels = grps.split(" & ")

            for label in labels:
                first = label.split("-")[0]
                second = label.split("-")[1]
            
                if first == "gender":
                    tmp_class.append(1)
                    will_append = True
                elif first == "race":
                    tmp_class.append(2)
                    will_append = True
                elif first == "religion":
                    tmp_class.append(3)
                    will_append = True                
                elif first == "politics":
                    tmp_class.append(5)
                    will_append = True
                elif first == "other":
                    if second == "age":
                        will_append = True
                        tmp_class.append(4)
                    elif second == "feminist":
                        will_append = True
                        tmp_class.append(1)
        if will_append == True:
            text_list.append(comment)
            class_list.append(tmp_class)

    # offensive true 고 group 없으면 넘어감 
    labeled_kold_df = pd.DataFrame({
        'text' : text_list,
        'class': class_list   
    })
    
    labeled_kold_df = labeled_kold_df[labeled_kold_df['class'].apply(lambda x: len(x) == 1)]
    labeled_kold_df['class'] = labeled_kold_df['class'].apply(lambda x: x[0])
    
    labeled_kold_df.reset_index(drop=True,inplace=True)
    return labeled_kold_df