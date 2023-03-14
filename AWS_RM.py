import numpy as np
import boto3
import io
import pandas as pd
from IPython.display import clear_output
import os
import csv

access_key = #Fill in this data
secret_access_key = # Fill in this data
endpoint = 'https://s3.us-east-2.amazonaws.com'  #Fill this in  
region = "us-east-2"

#Initialize the S3 client
s3 = boto3.client('s3', endpoint_url=endpoint,
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_access_key)


def start_model(project_arn, model_arn, version_name, min_inference_units):

    client=boto3.client('rekognition', region, aws_access_key_id = access_key,
        aws_secret_access_key = secret_access_key)

    try:
        # Start the model
        print('Starting model: ' + model_arn)
        response=client.start_project_version(ProjectVersionArn=model_arn, MinInferenceUnits=min_inference_units)
        # Wait for the model to be in the running state
        project_version_running_waiter = client.get_waiter('project_version_running')
        project_version_running_waiter.wait(ProjectArn=project_arn, VersionNames=[version_name])

        #Get the running status
        describe_response=client.describe_project_versions(ProjectArn=project_arn,
            VersionNames=[version_name])
        for model in describe_response['ProjectVersionDescriptions']:
            print("Status: " + model['Status'])
            print("Message: " + model['StatusMessage']) 
    except Exception as e:
        print(e)
        
    print('Done...')



def show_custom_labels(model,bucket,photo, min_confidence):
    client=boto3.client('rekognition', region, aws_access_key_id = access_key,
        aws_secret_access_key = secret_access_key)

    #Call DetectCustomLabels
    response = client.detect_custom_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        MinConfidence=min_confidence,
        ProjectVersionArn=model)


    return response['CustomLabels'][0]


def stop_model(model_arn):

    client=boto3.client('rekognition', region, aws_access_key_id = access_key,
        aws_secret_access_key = secret_access_key)

    print('Stopping model:' + model_arn)

    #Stop the model
    try:
        response=client.stop_project_version(ProjectVersionArn=model_arn)
        status=response['Status']
        print ('Status: ' + status)
    except Exception as e:  
        print(e)  

    print('Done...')
# def main():

#     bucket='torreon.fotos.frentes.predios'
#     photo='15_09_22/copiloto/0006d430-86e3-45d2-b52d-79e41b3c5a93.jpg'
#     model='arn:aws:rekognition:us-east-2:392104097481:project/FrentesTorreonComputerVision01/version/FrentesTorreonComputerVision01.2023-02-14T21.35.18/1676432120551'
#     min_confidence=70

#     label_count=show_custom_labels(model,bucket,photo, min_confidence)
#     #print("Custom labels detected: " + str(label_count))
#     print(label_count)


# if __name__ == "__main__":
#     main()
project_arn = 'arn:aws:rekognition:us-east-2:392104097481:project/FrentesTorreonComputerVision01/1676426890238'
model_arn = 'arn:aws:rekognition:us-east-2:392104097481:project/FrentesTorreonComputerVision01/version/FrentesTorreonComputerVision01.2023-02-14T21.35.18/1676432120551'
min_inference_units = 1 
version_name = 'FrentesTorreonComputerVision01.2023-02-14T21.35.18'
bucket ='torreon.fotos.frentes.predios'
min_confidence = 0


#start_model(project_arn, model_arn, version_name, min_inference_units)

batch = 'test' #input('Numero de division')

object_df=pd.read_csv(f'C:/Fotointrepretacion_Torreon/ML_results_ALL/divisiones/division_{batch}.csv', header=None)
object_list = object_df[0].tolist()

#result_df = pd.read_csv(f'/home/carolina/Documents/Fotointerpretacion_Torreon/test_results.txt', header=None)
results = []
i = 1
errores = 0
results = []
for object in object_list:
    try:
        temp_list = []
        print('Iteracion:',i)
        temp_dic = show_custom_labels(model_arn, bucket, object, min_confidence)
        print('Se clasifico')
        print(temp_dic)
        results.append([temp_dic['Name'],temp_dic['Confidence'],object])

        os.system('clear')
        i = i+1
    except:
        errores = errores + 1
        print('Numero de errores:',errores)
        continue


with open(f'C:/Fotointrepretacion_Torreon/ML_results_ALL/divisiones/result_{batch}.txt', 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    #write.writerow(fields)
    write.writerows(results)

stop_model(model_arn)


