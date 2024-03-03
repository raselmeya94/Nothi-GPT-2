import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the fine-tuned model and tokenizer
output_dir = './fine_tuned_bn_gpt2'
tokenizer = AutoTokenizer.from_pretrained(output_dir)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(output_dir)

# Define a text generation pipeline with the fine-tuned model
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Bengali digits convert into English digits

def bengali_to_english(bengali_number):
    # Dictionary mapping Bengali digits to English digits
    bengali_to_english = {'০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4', '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'}
    
    # Convert each Bengali digit to English digit
    english_number = ''.join(bengali_to_english.get(digit, digit) for digit in bengali_number)
    return english_number

def english_to_bengali(english_number):
    # Dictionary mapping English digits to Bengali digits
    english_to_bengali = {'0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪', '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'}
    
    # Convert each English digit to Bengali digit
    bengali_number = ''.join(english_to_bengali.get(digit, digit) for digit in str(english_number))
    return bengali_number
def calculate_sequence_length(input_text):
    initial_sequence_length=100
    # Define thresholds for short and long inputs
    short_threshold = 10
    long_threshold = 50

    # Calculate the length of the input text
    input_length = len(input_text.split())

    # Adjust sequence length based on input length
    if input_length < short_threshold:
        return initial_sequence_length  # Short input, use a smaller sequence length
    elif input_length > long_threshold:
        return initial_sequence_length*3  # Long input, use a larger sequence length
    else:
        return initial_sequence_length*2  # Default sequence length



# Define the Streamlit app
def main():
    #st.title("নথি-জিপিটি মডেল")
    st.markdown("<h1 style='text-align: center;'>নথি-জিপিটি মডেল</h1>", unsafe_allow_html=True)

    # User input for text prompt
    prompt = st.text_input("বাংলা টেক্সট প্রম্পট লিখুন:", value='', placeholder='এখানে লিখুন...')

    # User input for sequence number
    sequence_number_placeholder = "আউটপুট সংখ্যা (উদাহরণঃ ১)"
    sequence_number_bengali = st.text_input("আউটপুট সংখ্যা লিখুন:", value='', placeholder=sequence_number_placeholder)
    sequence_number = bengali_to_english(sequence_number_bengali)

    # User input for max length
    # max_length_placeholder = "আউটপুটের সর্বোচ্চ দৈর্ঘ্য (উদাহরণঃ ১০০)"
    # max_length_bengali = st.text_input("আউটপুটের সর্বোচ্চ দৈর্ঘ্য লিখুন:", value='', placeholder=max_length_placeholder)
    # max_length = bengali_to_english(max_length_bengali)

    # Generate text on button click
    if st.button("তৈরি করুন"):
        if not prompt:
            st.error("প্রম্পট ক্ষেত্র খালি রাখা যাবে না। দয়া করে একটি প্রম্পট লিখুন।")
        elif not sequence_number:
            st.error("আউটপুটের সর্বোচ্চ দৈর্ঘ্য ক্ষেত্র খালি রাখা যাবে না। দয়া করে একটি সর্বোচ্চ দৈর্ঘ্য লিখুন।")
        else:

            # Calculate sequence length dynamically based on input text
            sequence_length = calculate_sequence_length(prompt)
            # Generate text using the fine-tuned model
            generated_text = text_generator(prompt, max_length=int(sequence_length), num_return_sequences=int(sequence_number))

            # Display the generated text
            for i, text in enumerate(generated_text):
                st.text_area(f"উত্তর {english_to_bengali(i + 1)}:", value=text['generated_text'])

if __name__ == "__main__":
    main()
