from deep_translator import GoogleTranslator
import numpy as np
import time
from tqdm import tqdm


def translate_text(text, target_language, source_language='auto'):
    # Translate the text
    translation = GoogleTranslator(source=source_language, target=target_language).translate(text)
    return translation

def translate_with_retry(text, target_language, source_language, retries=3, delay=1):
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
    translated_values = []

    for text in tqdm(values, desc=f"Translating to {target_language}"):
        index=0
        missing_values = []
        result = None
        try:
            result = translate_with_retry(text, target_language, source_language)
            translated_values.append(result)
        except Exception as e:
            print(f"Error translating text: {text}")
            print(e)
            missing_values.append([text, index])

        time.sleep(0.5)
        index+=1        
      
    return np.array(translated_values), np.array(missing_values)

def main():
    # Load the data
    data = np.genfromtxt("./prompts/prompts_en.csv", delimiter=',', skip_header=0, dtype=str)  # Adjust if there is a header
    columns = data[:1] 
    values = data[1:10]

    # Define target languages
    target_languages = ["ja", "ko", "zh-CN", "zh-TW"]

    # Translate to target languages
    for lang in target_languages:
        # Translate values with progress tracking
        translated_values, missing_data = translate_with_progress(values.flatten(), lang, "en")
        translated_values = translated_values.reshape((translated_values.shape[0]//2,2))  # Reshape back to original shape

        # Stack columns and translated values
        translated_data= np.vstack((columns, translated_values))
        
        # Save to CSV
        np.savetxt(f"./prompts/prompts_{lang}.csv", translated_data, fmt='%s', delimiter=",")
        np.savetxt(f"./prompts/missing_prompts/{lang}.csv", missing_data, fmt='%s', delimiter=",")
        
        print(f"Done translating to {lang}")

if __name__ == "__main__":
    main()
    # data = np.genfromtxt("./prompts/prompts_en.csv", delimiter=',', skip_header=0, dtype=str)  # Adjust if there is a header
    # columns = data[:1] 
    # values = data[1:10]
    # translated_values = values.flatten()
    # translated_values = translated_values.reshape(values.shape)
    # print(translated_values)
    # print("AAAA")
    # print(values)
    # print("BBBB")
    # print(values.flatten())
    # assert np.array_equal(values, translated_values)
    # vectorized_translate_text = np.vectorize(translate_text)
    # print(vectorized_translate_text(["close-up photo of a young writer wearing ethnic clothes at home"], "ja"))
    # print(translate_text("close-up photo of a young writer wearing ethnic clothes at home", "ja"))
