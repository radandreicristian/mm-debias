from googletrans import Translator
import numpy as np
import time
from tqdm import tqdm

# Initialize the translator
translator = Translator()

def save_with_encoding(filename, data, delimiter=";"):
    with open(filename, "w", encoding="utf-8", newline='') as f:
        np.savetxt(f, data, fmt='%s', delimiter=delimiter)

def translate_text(text, target_language, source_language='auto'):
    """
    Translate the text from source_language to target_language
    text: Text to translate
    target_language: Target language code
    source_language: Source language code
    return: Translated text as a string
    """
    translation = translator.translate(text, src=source_language, dest=target_language)
    return translation.text

def translate_with_retry(text, target_language, source_language, retries=10, delay=1):
    """
    Translate the text with retries in case of failure.
    retries: Number of retries
    delay: Delay between retries in seconds
    return: Translated text as a string
    """
    for attempt in range(retries):
        try:
            # Attempt to translate the text
            result = translate_text(text, target_language, source_language)
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for text: {text}")
            print(e)
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                print(f"Failed to translate text after {retries} attempts.")
                raise  # Re-raise the exception after the final attempt


def translate_with_progress(values, target_language, source_language='auto'):
    """
    Translate the values to the target language with progress tracking.
    values: Array of text values to translate
    target_language: Target language code
    source_language: Source language
    return: Translated values as a numpy array
    """
    translated_values = []
    missing_values = []
    index=0
    for text in tqdm(values, desc=f"Translating to {target_language}"):
        result = None
        try:
            result = translate_with_retry(text, target_language, source_language)
            translated_values.append(result)
        except Exception as e:
            print(f"Error translating text: {text}")
            print(e)
            missing_values.append([text, index])
        time.sleep(0.20)
        index+=1        
      
    return np.array(translated_values), np.array(missing_values)

def main():
    # Load the data
    data = np.genfromtxt("./prompts/prompts_en.csv", delimiter=',', skip_header=0, dtype=str)  # Adjust if there is a header
    columns = data[:1] 
    values = data[1:]

    # Define target languages
    target_languages = ["de"]

    # Translate prompts to target languages
    for lang in target_languages:
        # Translate values with progress tracking
        translated_values, missing_data = translate_with_progress(values.flatten(), lang, "en")
        save_with_encoding(f"./prompts/linear_prompts/prompts_{lang}_linear.csv", translated_values)

        # If all values are translated, save the data in the original format
        if translated_values.size == values.size:
            translated_values = translated_values.reshape((translated_values.shape[0]//2,2))  # Reshape back to original shape
            translated_data= np.vstack((columns, translated_values))
            save_with_encoding(f"./prompts/prompts_{lang}.csv", translated_data)

        # Save missing data to a separate file, to run translation again and insert the values at the correct index
        if missing_data.size > 0:
            save_with_encoding(f"./prompts/missing_prompts/{lang}.csv", missing_data)
        
        print(f"Done translating to {lang}")

if __name__ == "__main__":
    main()
