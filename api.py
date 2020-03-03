# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:33:34 2020

@author: Maria
"""
from algorithms import importDataset, splitAndScale, ANNregression, randomForest, svr
from sklearn.preprocessing import MinMaxScaler
import numpy as np

###############################################
#              Choosing Dataset               #
###############################################
def selectDataset():
    print("Select a file to use:")
    print("1 - Regression Original 1437 rows")
    print("2 - Regression Balanced (83% deleted)")
    print("3 - Regression Encoded variables all 1437 rows")
    print("4 - Regression Encoded variables balanced (83% deleted)")
    print("5 - Regression no 365 days")
    print("6 - Regression only 365 days")
    print("7 - Regression only synthetic 3211 rows")
    print("8 - Regression synthetic plus 365 days")
    
    number = 0
    acceptedDataset = False
    while acceptedDataset is False:
        number = int(input("Select number to import dataset: "))
        if number > 0 and number < 9:
            acceptedDataset = True
        else:
            print("Invalid number, select a dataset by selecting its number (1 to 8)")
    
    choice = ""
    if number == 1: choice = "regAll.csv"
    if number == 2: choice = "regBalanced.csv"
    if number == 3: choice = "regEncoded.csv"
    if number == 4: choice = "regEncodedBalanced.csv"
    if number == 5: choice = "regNo365.csv"
    if number == 6: choice = "regOnly365.csv"
    if number == 7: choice = "regSynthetic.csv"
    if number == 8: choice = "regSyntheticWith365.csv"
    
    print("dataset chosen:", choice)
    # Import the Dataset and separate X and y
    X_before, y_before = importDataset(choice)
    # Split the dataset
    X_train, X_test, y_train, y_test = splitAndScale(X_before, y_before)
    
    return X_before, y_before, X_train, X_test, y_train, y_test
    

###############################################
#            Choosing Algorithm               #
###############################################
def chooseAlgorithm(X_before, X_train, X_test, y_train, y_test):
    print("\n Choose an algorithm to run on the dataset:")
    print("1 - Artificial Neural Network")
    print("2 - Random Forest")
    print("3 - Support Vector Regression")
    
    number = 0
    acceptedAlgorithm = False
    while acceptedAlgorithm is False:
        number = int(input("Select number of algorithm to run: "))
        if number > 0 and number < 4:
            acceptedAlgorithm = True
        else:
            print("Invalid number, select an algorithm by selecting its number (1 to 3)")
            
    if number == 1: 
        regressor = ANNregression(X_train, y_train, X_test, y_test)
        return regressor
    if number == 2: 
        rfModel = randomForest(X_train, y_train, X_test, y_test, X_before)
        return rfModel
    if number == 3: 
        svrModel = svr(X_train, y_train, X_test, y_test)
        return svrModel
    
def ask():
    print("\nWhat do you want to do now?")
    print("1 - Choose another dataset to train or a different model")
    print("2 - Predict from manual input of donor and recipient variables")
    print("3 - exit")
    keepWorkin = input("> ")
    return keepWorkin
    
def nextSteps(choice):
    
    if int(choice) == 1: 
        print("1")
        X_before, y_before, X_train, X_test, y_train, y_test = selectDataset()
        model = chooseAlgorithm(X_before, X_train, X_test, y_train, y_test)
    if int(choice) == 2:
        try:
            print("Insert recipient's values: ")
            while True:
                age = int(input("- Age: "))
                if age > 10 and age < 80: break
                else: print("Invalid value, must be between 10 and 80")
                
            while True:
                gender = int(input("- Gender (1 - Male, 0 - Female): "))
                if gender == 1 or gender == 0: break
                else: print("Invalid value, must be 1 for male or 0 for female")
                
            while True:
                bmibasal = int(input("- Body-Mass Index (in kg/m2): "))
                if bmibasal > 12 and bmibasal < 76: break
                else: print("Invalid value, must be between 13 and 75")
                
            while True:
                diabetesPreTx = int(input("- diabetes (1 - yes, 0 - no): "))
                if diabetesPreTx == 1 or diabetesPreTx == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                htabasal = int(input("- Arterial hypertension (1 - yes, 0 - no): "))
                if htabasal == 1 or htabasal == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                dialisis = int(input("- Dialysis requirement pre-transplant (1 - yes, 0 - no): "))
                if dialisis == 1 or dialisis == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                etiologiaprincipal = int(input("- Etiology justifying transplant need:\n\t\t0 - Virus C cirrhosis\n\t\t1- Alcohol cirrhosis\n\t\t2 - Virus B cirrhosis\n\t\t3 - Fulminant hepatic failure\n\t\t4 - Primary biliary cirrhosis\n\t\t5 - Primary sclerosing cholangitis\n\t\t6 - Others\n> "))
                if etiologiaprincipal < 7: break
                else: print("Invalid value, must be a number between 0 and 6")
            
            while True:
                trombosisportal = int(input("- Portal thrombosis:\n\t\t0 - No portal thrombosis\n\t\t1 - Partial\n\t\t2 - Complete\n> "))
                if trombosisportal < 3: break
                else: print("Invalid value, must be a number between 0 and 2")
            
            while True:
                tiempolistaespera = int(input("Waiting list time (in days): "))
                if tiempolistaespera > 0 and tiempolistaespera < 2000: break
                else: print("Invalid value, must be between 1 and 2000")
            
            while True:
                meldinclusion = int(input("- MELD score at waiting list inclusion: "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 50")
            
            while True:
                meldtx = int(input("- MELD at transplant time: "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 60")
            
            while True:
                tips = int(input("- TIPS at transplant (1 - yes, 0 - no): "))
                if tips == 1 or tips == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                sindromehepatorrenal = int(input("- Hepatorrenal syndrome (1 - yes, 0 - no): "))
                if sindromehepatorrenal == 1 or sindromehepatorrenal == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                apcirugiaabdosuperior = int(input("- History of previous upper abdominal surgery (1 - yes, 0 - no): "))
                if apcirugiaabdosuperior == 1 or apcirugiaabdosuperior == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                sfiptx = int(input("- Cytomegalovirus (1 - yes, 0 - no): "))
                if sfiptx == 1 or sfiptx == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                cmvbasal = int(input("- Pre-transplant status performance:\n\t\t0 - At home\n\t\t1 - Hospitalised\n\t\t2 - Hospitalised in ICU\n\t\t3 - Hospitalised in ICU with mechanical ventilation\n> "))
                if cmvbasal < 4: break
                else: print("Invalid value, must be a number between 0 and 3")
                
            
            print("\n Insert donor's values:")
            while True:
                edaddon = int(input("- Age: "))
                if edaddon > 10 and edaddon < 80: break
                else: print("Invalid value, must be between 10 and 80")
                
            while True:
                sexodon = int(input("- Gender (1 - Male, 0 - Female): "))
                if sexodon == 1 or sexodon == 0: break
                else: print("Invalid value, must be 1 for male or 0 for female")
                
            while True:
                bmiestdon = int(input("- Body-Mass Index (in kg/m2): "))
                if bmiestdon > 12 and bmiestdon < 76: break
                else: print("Invalid value, must be between 13 and 75")
                
            while True:
                diabetesmelitusdon = int(input("- diabetes (1 - yes, 0 - no): "))
                if diabetesmelitusdon == 1 or diabetesmelitusdon == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                htadon = int(input("- Arterial hypertension (1 - yes, 0 - no): "))
                if htadon == 1 or htadon == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                causaexitus = int(input("- Cause of death:\n\t\t0 - Brain trauma\n\t\t1 - Cerebral vascular accident (CVA)\n\t\t2 - Anoxia\n\t\t3 - Deceased vascular after cardiac arrest\n\t\t4 - Others\n> "))
            
            while True:
                diasuci = int(input("- Hospitalised length in ICU (days): "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 50")
            
            while True:
                hipotension = int(input("- Hypotension episodes (1 - yes, 0 - no): "))
                if hipotension == 1 or hipotension == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                inotropos = int(input("- High inotropic drug use (1 - yes, 0 - no): "))
                if inotropos == 1 or inotropos == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                creatinina = int(input("- Creatinine plasma level (in mg/dl): "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 50")
            
            while True:
                na = int(input("- Sodium plasma level (in mEq/l): "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 50")
            
            while True:
                ast = int(input("- Aspartate transaminase level: (in UI/l): "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 50")
            
            while True:
                alt = int(input("- Alanine aminotransferase plasma level (in UI/l): "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 50")
            
            while True:
                bit = int(input("- Total bilirubin (in mg/dl): "))
                if meldinclusion > 0 and meldinclusion < 50: break
                else: print("Invalid value, must be between 1 and 50")
            
            while True:
                antihbc = int(input("- Hepatitis B (1 - yes, 0 - no): "))
                if antihbc == 1 or antihbc == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                vhc = int(input("- Hepatitis C (1 - yes, 0 - no): "))
                if vhc == 1 or vhc == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                cmvdon = int(input("- Cytomegalovirus (1 - yes, 0 - no): "))
                if cmvdon == 1 or cmvdon == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            
            print("\nInsert transplant info:")
            while True:
                multiorganico = int(input("- Multi-organ harvesting (1 - yes, 0 - no): "))
                if multiorganico == 1 or multiorganico == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                txcombinado = int(input("- Combined transplant (1 - yes, 0 - no): "))
                if txcombinado == 1 or txcombinado == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            while True:
                injertocompletoparcial = int(input("- Complete or partial graft (1 - complete, 0 - partial): "))
                if injertocompletoparcial == 1 or injertocompletoparcial == 0: break
                else: print("Invalid value, must be 0 for partial or 1 for complete")
            
            while True:
                tiempoisquemiafria = int(input("- Cold ischemia time:\n\t\t0 - Less than 6 hours\n\t\t1 - Between 6 and 12 hours\n\t\t2 - More than 6 hours\n> "))
            
            while True:
                compatibilidadabo = int(input("- AB0 compatible transplant (1 - yes, 0 - no): "))
                if compatibilidadabo == 1 or compatibilidadabo == 0: break
                else: print("Invalid value, must be 0 for no or 1 for yes")
            
            
            to_predict = [age, gender, bmibasal, diabetesPreTx, htabasal, dialisis,
                      etiologiaprincipal, trombosisportal, tiempolistaespera,
                      meldinclusion, meldtx, tips, sindromehepatorrenal,
                      apcirugiaabdosuperior, sfiptx, cmvbasal,
                      edaddon, sexodon, bmiestdon, diabetesmelitusdon, htadon,
                      causaexitus, diasuci, hipotension, inotropos, creatinina,
                      na, ast, alt, bit, antihbc, vhc, cmvdon,
                      multiorganico, txcombinado, injertocompletoparcial,
                      tiempoisquemiafria, compatibilidadabo
                     ]
        
            print(to_predict)
        except ValueError: print("invalid input")
        
        
#        scaler = MinMaxScaler()
#        
#        new_prediction = model.predict(scaler.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

    if int(choice) == 3:
        return True
    return False
        
def main():
    print("\n\tWELCOME TO THE LIVER TRANSPLANT DONOR-RECIPIENT MATCH PREDICTOR\n")
    
    X_before, y_before, X_train, X_test, y_train, y_test = selectDataset()
    model = chooseAlgorithm(X_before, X_train, X_test, y_train, y_test)
    print(model)

    finished = False
    while finished == False:
        x = ask()
        four = nextSteps(x)
        if four == True:
            print("\n\tBYE BYE")
            break
    

    
if __name__ == "__main__":
    main()



