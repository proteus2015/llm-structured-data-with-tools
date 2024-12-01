# llm-structucted-data-with-tools (series 2)
This is series 2 of LLM + RAG (series 1 is in https://github.com/proteus2015/llm-on-structured-unstructured-data/blob/main/README.md#llm-on-structed-unstrcted-data-series-1) focusing on LLM + RAG for structured data. A tool plotly is added in langchain LLM+ RAG chain to display output from LLM result. 
Architecture:

![image](https://github.com/user-attachments/assets/b4d20b98-12ca-4225-b1bf-b33cba572a5f)

Packages and condistions needed:
1. creat an account and an OpenAI api key in (https://platform.openai.com/api-keys) , and save they key in your local environment with name OPENAI_API_KEY. In the code, the key will be retrieved like this:

    import os
   
    my_openai_api_key = os.getenv("OPENAI_API_KEY")

    from langchain_openai import ChatOpenAI
   
    llm = ChatOpenAI(
          temperature=0, model="gpt-3.5-turbo", openai_api_key=my_openai_api_key, streaming=True
   )
   
2. go to Openai your profile/billing, and deposit money there. Each OpenAI call costs ~1 cent in my case (with 10M data maximum), so I only deposited $10 there.
   
3. Install necessary packages

   pip install langchain   
   pip install streamlit   
   pip install pandas
   pip install plotly==5.24.1
   
5. How to run:
   4.1 For csv and any structured data pandas can process, you only need to run dataset.server under folder vectoring, for example, in pycharm Terminal (community version is good enough):
 
       cd vectorizing
   
       streamlit run .\dataset-serve-with-tools.py
   

   A new URL will pop up. Current codes support upload 1 to 2 csv files; then raise questions on the uploaded files in the same URL. An example is shown below:

<img width="536" alt="numeric tabular data result-8  draw line chart with plotly as Tool-1" src="https://github.com/user-attachments/assets/1ab44ceb-ecde-41c9-a0ad-aeffc0a71c21">
<img width="935" alt="numeric tabular data result-8  draw line chart with plotly as Tool-2" src="https://github.com/user-attachments/assets/3631fb41-7961-4350-8125-73a690ca59bd">

    

              
 
          
  
