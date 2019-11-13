To Train a model, follow below steps:
1. Run below script to clean the data for any "nan" values:
   clean_data.py
   sample command: python clean_data.py --folder <folder_location> 

2. Extract fearures using below script:
   extract_features.py
   sample command: python extract_features.py --mealfolder <folder with cleaned meal files> --nomealfolder <folder with cleaned no_meal files>

3. Run Training model:
   model.py
   sample command (same folder as above): python model.py --meal_folder <> --no_meal_folder <>
   
