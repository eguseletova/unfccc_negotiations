import pdfplumber
import pandas as pd
import re
import networkx as nx
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import unicodedata


## country dictionaries
country_aliases = {
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "U.S.": "United States",
    "US": "United States",
    "USA": "United States",
    "South Korea": "Republic of Korea",
    "North Korea": "Democratic People's Republic of Korea",
    "DR of the Congo": "Democratic Republic of the Congo",
    "DRC": "Democratic Republic of the Congo",
    "Republic of Congo": "Congo",
    "Czech Republic": "Czechia",
    "Iran (Islamic Republic of)": "Iran",
    "Viet Nam": "Vietnam",
    "Russia": "Russian Federation",
    "Vatican": "Holy See",
    "Syrian Arab Republic": "Syria",
    "State of Palestine": "Palestine",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Turkey": "Türkiye",
    "Timor Leste": "Timor-Leste"
}

## countries and negotiation blocs
countries = list(set(country_aliases.values()) | set([
    "Afghanistan", "Algeria", "Albania", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize",
    "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo", "Cook Islands", "Costa Rica", "Côte d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czechia",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea",
    "Eritrea", "Eswatini", "Estonia", "Ethiopia", "EU", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
    "Honduras", "Hungary", "Holy See", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos",
    "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
    "Micronesia", "Moldova", "Monaco", "Montenegro", "Mongolia", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru",
    "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "North Macedonia",
    "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russian Federation", "Rwanda", "Samoa", "San Marino",
    "Saudi Arabia", "Sao Tome and Principe", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Türkiye", "Turkmenistan", "Tuvalu",
    "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]))

negotiation_blocs = {
    "AFRICAN GROUP": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", 
                      "Cameroon", "Central African Republic", "Chad", "Comoros", "Democratic Republic of the Congo", 
                      "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", 
                      "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Côte d'Ivoire", "Kenya", "Lesotho", "Liberia", 
                      "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", 
                      "Mozambique", "Namibia", "Niger", "Nigeria", "Republic of the Congo", 
                      "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", 
                      "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", 
                      "Uganda", "Zambia", "Zimbabwe"],

    "AILAC": ["Chile", "Colombia", "Costa Rica", "Guatemala", "Honduras", "Panama", "Peru"],

    "ALBA": ["Antigua and Barbuda", "Bolivia", "Cuba", "Dominica", "Grenada", "Nicaragua", "Saint Kitts and Nevis", 
             "Saint Lucia", "Saint Vincent and the Grenadines", "Venezuela"],

    "AOSIS": ["Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Cabo Verde", "Comoros", "Cook Islands", 
              "Cuba", "Dominica", "Dominican Republic", "Micronesia", "Fiji", 
              "Grenada", "Guinea Bissau", "Guyana", "Haiti", "Jamaica", "Kiribati", "Maldives", "Mauritius", 
              "Nauru", "Niue", "Palau", "Papua New Guinea", "Marshall Islands", "Saint Kitts and Nevis", 
              "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "Sao Tome and Principe", 
              "Seychelles", "Singapore", "Solomon Islands", "Suriname", "Timor Leste", "Tonga", 
              "Trinidad and Tobago", "Tuvalu", "Vanuatu"],

    "ARAB GROUP": ["Algeria", "Bahrain", "Comoros", "Djibouti", "Egypt", "Iraq", "Jordan", "Kuwait", "Lebanon", 
                   "Libya", "Morocco", "Mauritania", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Somalia", 
                   "Sudan", "Syria", "Tunisia", "United Arab Emirates", "Yemen"],

    "BASIC": ["Brazil", "South Africa", "India", "China"],

    "CfRN": ["Argentina", "Bangladesh", "Belize", "Bolivia", "Botswana", "Brazil", "Cambodia", "Cameroon", 
             "Central African Republic", "China", "Costa Rica", "Democratic Republic of the Congo", "DR of the Congo",
             "Dominica", "Dominican Republic", "Ecuador", "Equatorial Guinea", "Fiji", "Gabon", "Ghana",  
             "Guatemala", "Guyana", "Honduras", "India", "Jamaica", "Kenya", "Laos", 
             "Lesotho", "Liberia", "Madagascar", "Malawi", "Malaysia", "Mali", "Mozambique", "Namibia", "Nicaragua", 
             "Nigeria", "Pakistan", "Papua New Guinea", "Paraguay", "Republic of Congo", "Indonesia", 
             "Saint Lucia", "Samoa", "Sierra Leone", "Singapore", "Solomon Islands", "South Africa", "Sudan", 
             "Suriname", "Thailand", "Uganda", "Uruguay", "Vanuatu", "Vietnam", "Zambia"],

    "EIG": ["Mexico", "Liechtenstein", "Monaco", "Republic of Korea", "Switzerland", "Georgia"],

    "EU": ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia", 
           "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", 
           "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"],

    "G77/CHINA": ["Afghanistan", "Algeria", "Angola", "Antigua and Barbuda", "Argentina", "Azerbaijan", "Bahamas", "Bahrain",
                  "Bangladesh","Barbados","Belize","Benin","Bhutan","Bolivia (Plurinational State of)","Botswana","Brazil",
                  "Brunei","Burkina Faso","Burundi","Cabo Verde","Cambodia","Cameroon","Central African Republic",
                  "Chad","Chile","China","Colombia","Comoros","Congo","Costa Rica","Côte d'Ivoire","Cuba","Democratic People's Republic of Korea",
                  "Democratic Republic of the Congo","Djibouti","Dominica","Dominican Republic","Ecuador","Egypt","El Salvador",
                  "Equatorial Guinea","Eritrea","Eswatini","Ethiopia","Fiji","Gabon","Gambia","Ghana","Grenada",
                  "Guatemala","Guinea","Guinea-Bissau","Guyana","Haiti","Honduras","India","Indonesia",
                  "Iran","Iraq","Jamaica","Jordan","Kenya","Kiribati","Kuwait","Lao People's Democratic Republic",
                  "Lebanon","Lesotho","Liberia","Libya","Madagascar","Malawi","Malaysia","Maldives","Mali","Marshall Islands","Mauritania",
                  "Mauritius","Micronesia","Mongolia","Morocco","Mozambique","Myanmar","Namibia","Nauru",
                  "Nepal","Nicaragua","Niger","Nigeria","Oman","Pakistan","Panama","Papua New Guinea","Paraguay",
                  "Peru","Philippines","Qatar","Rwanda","Saint Kitts and Nevis","Saint Lucia",
                  "Saint Vincent and the Grenadines","Samoa","Sao Tome and Principe","Saudi Arabia","Senegal",
                  "Seychelles","Sierra Leone","Singapore","Solomon Islands","Somalia","South Africa","South Sudan",
                  "Sri Lanka","Palestine","Sudan","Suriname","Syria","Tajikistan","Thailand",
                  "Timor-Leste","Togo","Tonga","Trinidad and Tobago","Tunisia","Turkmenistan","Uganda","United Arab Emirates",
                  "Tanzania","Uruguay","Vanuatu","Venezuela","Viet Nam","Yemen","Zambia","Zimbabwe"],

    "LDCs": ["Afghanistan", "Angola", "Bangladesh", "Benin", "Burkina Faso", "Burundi", "Cambodia", "Central African Republic",
             "Chad", "Comoros", "the Democratic Republic of the Congo", "Djibouti", "Eritrea", "Ethiopia", "Gambia", "Guinea",
             "Guinea-Bissau", "Haiti", "Kiribati", "Laos", "Lesotho", "Liberia", "Madagascar",
             "Malawi", "Mali", "Mauritania", "Mozambique", "Myanmar", "Niger", "Nepal", "Rwanda", "Sao Tome and Principe", "Senegal",
             "Sierra Leone", "Solomon Islands", "Somalia", "South Sudan", "Sudan", "Tanzania", "Timor-Leste", "Togo", "Tuvalu",
             "Uganda", "Yemen", "Zambia"],

    "LLDCs": ["Afghanistan", "Armenia", "Azerbaijan", "Bhutan", "Bolivia", "Botswana", "Burkina Faso", "Burundi", "Central African Republic",
              "Chad", "Eswatini", "Ethiopia", "Kazakhstan", "Kyrgyzstan", "Laos", "Lesotho", "North Macedonia",
              "Malawi", "Mali", "Mongolia", "Nepal", "Niger", "Paraguay", "Moldova", "Rwanda", "South Sudan", "Tajikistan", "Turkmenistan", "Uganda",
              "Uzbekistan", "Zambia", "Zimbabwe"],

    "LMDCs": ["Algeria", "Bangladesh", "Bolivia", "China", "Cuba", "Ecuador", "Egypt", "El Salvador", "India", "Indonesia", "Iran", "Iraq", "Jordan",
              "Kuwait", "Malaysia", "Mali", "Nicaragua", "Pakistan", "Saudi Arabia", "Sri Lanka", "Sudan", "Syria", "Venezuela", "Vietnam"],

    "UMBRELLA GROUP": ["Australia", "Canada", "Iceland", "Israel", "Japan", "Kazakhstan", "New Zealand", "Norway", "Ukraine", "US", 
                       "Belarus", "Russian Federation"]
}

# Loading & extracting data from pdfs
def clean_text(text):
    if text:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        for alias, standard in country_aliases.items():
            text = re.sub(rf"\b{re.escape(alias)}\b", standard, text, flags=re.IGNORECASE)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract text from PDFs with cleaning
def extract_text_from_pdfs(file_paths):
    extracted_texts = []
    for file in file_paths:
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            text = clean_text(text)
            extracted_texts.append(text)
    return extracted_texts

## ENB & UNFCCC text data
## this code uses example text modelled on the Earth News Bulletin reports for copyright reasons
enb_files = ["cop28_day1_sample.pdf"]
enb_texts = extract_text_from_pdfs(enb_files)
full_enb_text = " ".join(enb_texts)

# official UNFCCC files, in this case cop28
unfccc_files = ["cop28a.pdf", "cop28b.pdf"]
unfccc_texts = extract_text_from_pdfs(unfccc_files)
full_unfccc_text = " ".join(unfccc_texts)

sentences = re.split(r' *[\.\?!][\'"\)\]]* *', full_enb_text)


# country influence
adjusted_influence = {country: 0 for country in countries}

alignment_keywords = ["supported by", "supported", "welcomed", "agreed", "emphasized", "stressed", "highlighted", "underlined"]
opposition_keywords = ["opposed", "rejected", "criticized", "disagreed", "questioned", "opposed by"]

country_relations = Counter()

for sentence in sentences:
    speaker_match = re.search(r"(\b[A-Z][a-z]+\b), for (?:the )?([A-Z]+[\w\s&]+),", sentence)

    if speaker_match:
        speaker = speaker_match.group(1)  
        bloc = speaker_match.group(2)  

        if speaker not in adjusted_influence:
            adjusted_influence[speaker] = 0
        
        adjusted_influence[speaker] += 1 

        for member in negotiation_blocs.get(bloc, []):
            if member not in adjusted_influence:
                adjusted_influence[member] = 0  
            if member != speaker:
                adjusted_influence[member] += 0.01  

    else:
        for country in countries:
            if re.search(rf"\b{re.escape(country)}\b (proposed|stressed|highlighted|argued|said|noted|explained|urged|requested|emphasized|supported|welcomed|opposed|criticized)", sentence, re.IGNORECASE):
                if country not in adjusted_influence:
                    adjusted_influence[country] = 0

                adjusted_influence[country] += 1.0  


    mentioned_countries = [country for country in countries if re.search(rf"\b{re.escape(country)}\b", sentence, re.IGNORECASE)]
    alignment = any(word in sentence.lower() for word in alignment_keywords)
    opposition = any(word in sentence.lower() for word in opposition_keywords)

    for pair in combinations(mentioned_countries, 2):
        if alignment:
            country_relations[pair] += 1
        elif opposition:
            country_relations[pair] -= 1


## network analysis
G_signed = nx.Graph()
for (country1, country2), weight in country_relations.items():
    G_signed.add_edge(country1, country2, weight=weight)


max_influence = max(adjusted_influence.values(), default=1)
normalized_influence = {country: influence / max_influence for country, influence in adjusted_influence.items()}

# LDA topic modelling and text similarity
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_enb = vectorizer.fit_transform(enb_texts)
X_unfccc = vectorizer.transform(unfccc_texts)

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X_enb)


enb_topics = lda.transform(X_enb)
unfccc_topics = lda.transform(X_unfccc)

## opposition strength
opposition_strength = {country: 0 for country in countries}
for (country1, country2), weight in country_relations.items():
    if weight < 0:
        opposition_strength[country1] += abs(weight)
        opposition_strength[country2] += abs(weight)

max_opposition = max(opposition_strength.values(), default=1)
normalized_opposition = {country: strength / max_opposition for country, strength in opposition_strength.items()}

## final scores
final_success_scores = {}

for country in countries:
    country_sentences = [sentence for sentence in sentences if re.search(rf"\b{re.escape(country)}\b", sentence, re.IGNORECASE)]
    country_text = " ".join(country_sentences)

    if country_text:
        X_country = vectorizer.transform([country_text])
        country_topics = lda.transform(X_country)
        similarity = np.mean(cosine_similarity(country_topics, unfccc_topics))
    else:
        similarity = 0

    centrality_score = nx.degree_centrality(G_signed).get(country, 0)
    influence_score = normalized_influence.get(country, 0)
    opposition_factor = normalized_opposition.get(country, 0)

    success_score = 0.4 * similarity + 0.1 * influence_score + 0.3 * centrality_score + 0.2 * opposition_factor
    final_success_scores[country] = success_score


final_success_df = pd.DataFrame.from_dict(final_success_scores, orient="index", columns=["Final Success Score"])
final_success_df = final_success_df.sort_values(by="Final Success Score", ascending=False)

final_success_df.rename(columns={"Final Success Score": "cop28", "index": "country_name"}, inplace=True)

final_success_df.to_csv("success_scores.csv")


# print(final_success_df)

## for component validation
subcomponents_data = {
    "country_name": [],
    "text_similarity": [],
    "influence_score": [],
    "centrality_score": [],
    "opposition_factor": []
}

for country in countries:
    country_sentences = [sentence for sentence in sentences if re.search(rf'\b{re.escape(country)}\b', sentence, re.IGNORECASE)]
    country_text = " ".join(country_sentences)

    if country_text:
        X_country = vectorizer.transform([country_text])
        country_topics = lda.transform(X_country)
        similarity = np.mean(cosine_similarity(country_topics, unfccc_topics))
    else:
        similarity = 0

    subcomponents_data["country_name"].append(country)
    subcomponents_data["text_similarity"].append(similarity)
    subcomponents_data["influence_score"].append(normalized_influence.get(country, 0))
    subcomponents_data["centrality_score"].append(nx.degree_centrality(G_signed).get(country, 0))
    subcomponents_data["opposition_factor"].append(normalized_opposition.get(country, 0))

subcomponents_df = pd.DataFrame(subcomponents_data)

subcomponents_df.to_csv("success_components.csv", index=False)
