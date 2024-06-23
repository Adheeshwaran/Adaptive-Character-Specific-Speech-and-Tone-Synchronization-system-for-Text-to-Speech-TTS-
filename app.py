from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>story</title>
    </head>
    <style>
    body{
        box-sizing: border-box;
        margin: 0;
        font-family: sans-serif;
        font-size: medium;
    }
    h1{
        text-align: center;
        font-family: sans-serif;
        width: auto;
        height: 50px;
        padding: 22.45px;
        margin: 0;
        background-color: black;
        color: white;
    }
    form{
        display: flex;
        flex-direction: column;
        align-items: center;
        width: auto;
        height: 100px;
        gap: 1em;
        margin-top: 22.45px;
        padding: 20px;
        padding-bottom: 0;
    }
    input{
        background-color: black;
        color: white;
    }
    button{
        background-color: black;
        color: white;
        font-size: medium;
        padding: 5px;
        border-radius: .5px;
        cursor: pointer;
    }
    #div{
        display: grid;
        grid-template-columns: auto auto;
        padding: 10px;
        gap: .5rem;
    }
    #div div{
        background-color: #46a6f5;
        text-align: center;
    }
    h3{
    text-align:center
    }
    span{
    position: relative;
    color: rgba(0, 0, 0, .3);
    font-size: 2.5em;
    margin: 0 40%;
    }
    span:before {
        content: attr(data-text);
        position: absolute;
        overflow: hidden;
        max-width: 7em;
        white-space: nowrap;
        color: #46a6f5;
        animation: loading 8s linear infinite;
    }
    @keyframes loading {
        0% {
            max-width: 0;
        }
    }
    #sp{
    display: none;
    }
    .none{
    display: none
    }
</style>
    <body>
        <h1>Story Teller</h1>

        <form action="" onsubmit="sendMessage(event)">
            <input type="file" name="book" id="book">
            <button id="btn">convert</button>
        </form>
        <h3 id="h3"></h3>
<span data-text="converting..." id="sp">converting...</span>
        <div id="div"></div>
        

        
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var div = document.createElement('div')
                var h2 = document.createElement("h2")
                h2.innerHTML=event.data
                div.appendChild(h2);
                var audioElement = document.createElement("audio");
                audioElement.src="http://localhost:5500/chunks1/"+event.data+".mp3"
                audioElement.controls=true
                div.appendChild(audioElement)
                div.className=event.data;
                document.getElementById("div").appendChild(div);
                document.getElementById("sp").style.display=event.data
                
                
            };
            function sendMessage(event) {
                var f = document.getElementById("book")
                document.getElementById("h3").innerHTML=f.files[0].name
                ws.send(f.files[0].name)
        document.getElementById("sp").style.display="block"

                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        
        import pdfplumber
        import nltk 
        nltk.download('wordnet')
        nltk.download('names')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

        async def _main(pdf_path):

            def pdf_to_markdown(pdf_path):
                # Open the PDF file at the given path
                with pdfplumber.open(pdf_path) as pdf:
                    markdown_content = ""
                    # Loop through each page in the PDF
                    for page in pdf.pages:
                        # Extract text from each page
                        text = page.extract_text()
                        if text:
                            # Format the text with basic Markdown: double newline for new paragraphs
                            markdown_page = text.replace('\n', '\n\n')
                            # Add a separator line between pages
                            markdown_content += markdown_page + '\n\n---\n\n'

                    return markdown_content

            pdf_path = f"D:/69/pdf_audio_web/{pdf_path}"
            markdown_text = pdf_to_markdown(pdf_path)
            # print(markdown_text)
            print(len(markdown_text))
            # Print the extracted and formatted text
            import re

            def markdown_to_plain_text(markdown_text):
                # Remove Markdown URL syntax ([text](link)) and keep only the text
                text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_text)

                # Remove Markdown formatting for bold and italic text
                text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold with **
                text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic with *
                text = re.sub(r'\_\_([^_]+)\_\_', r'\1', text)  # Bold with __
                text = re.sub(r'\_([^_]+)\_', r'\1', text)      # Italic with _

                # Remove Markdown headers, list items, and blockquote symbols
                text = re.sub(r'#+\s?', '', text)  # Headers
                text = re.sub(r'-\s?', '', text)   # List items
                text = re.sub(r'>\s?', '', text)   # Blockquotes

                return text

            plain_text = markdown_to_plain_text(markdown_text)
            # print(plain_text)
            print(len(plain_text))# Printing the converted plain text


            cleaned_text = plain_text.replace("artifact", "")

            # Printing the cleaned text to verify the changes
            # print(cleaned_text)
            print(len(cleaned_text))


            def split_text(text, max_chunk_size=2114):
                chunks = []  # List to hold the chunks of text
                current_chunk = ""  # String to build the current chunk

                # Split the text into sentences and iterate through them
                for sentence in text.split('.'):
                    sentence = sentence.strip()  # Remove leading/trailing whitespaces
                    if not sentence:
                        continue  # Skip empty sentences

                    # Check if adding the sentence would exceed the max chunk size
                    if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                        current_chunk += sentence + "."  # Add sentence to current chunk
                    else:
                        chunks.append(current_chunk)  # Add the current chunk to the list
                        current_chunk = sentence + "."  # Start a new chunk

                # Add the last chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)

                return chunks
            chunks = split_text(plain_text)

            # Printing each chunk with its number
            # for i, chunk in enumerate(chunks, 1):
            #     print(f"Chunk {i}:\n{chunk}\n---\n")


            print("chunk_length: ",len(chunks))

            for i in chunks[:1]:
                print("chunk_type: ",type(i))

            import nltk
            from nameparser.parser import HumanName
            from nltk.corpus import wordnet
            import random 
            from nltk.corpus import names 
            import nltk 
            from pathlib import Path
            import openai

            def text_to_speech_female_1(input_text, output_file, model="tts-1-hd", voice="nova"):
                # Initialize the OpenAI client
                client = openai.OpenAI(api_key = "")

                # Make a request to OpenAI's Audio API with the given text, model, and voice
                response = client.audio.speech.create(
                    model=model,      # Model for text-to-speech quality
                    voice=voice,      # Voice type
                    input=input_text  # The text to be converted into speech
                )

                # Define the path for the output audio file
                speech_file_path = Path(output_file)

                # Stream the audio response to the specified file
                response.stream_to_file(speech_file_path)

                # Print confirmation message after saving the audio file
                print(f"Audio saved to {speech_file_path}")


            def text_to_speech_female_2(input_text, output_file, model="tts-1-hd", voice="shimmer"):
                # Initialize the OpenAI client
                client = openai.OpenAI(api_key = "")

                # Make a request to OpenAI's Audio API with the given text, model, and voice
                response = client.audio.speech.create(
                    model=model,      # Model for text-to-speech quality
                    voice=voice,      # Voice type
                    input=input_text  # The text to be converted into speech
                )

                # Define the path for the output audio file
                speech_file_path = Path(output_file)

                # Stream the audio response to the specified file
                response.stream_to_file(speech_file_path)

                # Print confirmation message after saving the audio file
                print(f"Audio saved to {speech_file_path}")





            def text_to_speech_male_1(input_text, output_file, model="tts-1-hd", voice="fable"):
                # Initialize the OpenAI client
                client = openai.OpenAI(api_key = "")

                # Make a request to OpenAI's Audio API with the given text, model, and voice
                response = client.audio.speech.create(
                    model=model,      # Model for text-to-speech quality
                    voice=voice,      # Voice type
                    input=input_text  # The text to be converted into speech
                )

                # Define the path for the output audio file
                speech_file_path = Path(output_file)

                # Stream the audio response to the specified file
                response.stream_to_file(speech_file_path)

                # Print confirmation message after saving the audio file
                print(f"Audio saved to {speech_file_path}")


            def text_to_speech_male_2(input_text, output_file, model="tts-1-hd", voice="onyx"):
                # Initialize the OpenAI client
                client = openai.OpenAI(api_key = "")

                # Make a request to OpenAI's Audio API with the given text, model, and voice
                response = client.audio.speech.create(
                    model=model,      # Model for text-to-speech quality
                    voice=voice,      # Voice type
                    input=input_text  # The text to be converted into speech
                )

                # Define the path for the output audio file
                speech_file_path = Path(output_file)

                # Stream the audio response to the specified file
                response.stream_to_file(speech_file_path)

                # Print confirmation message after saving the audio file
                print(f"Audio saved to {speech_file_path}")

            import os
            from pydub import AudioSegment
            import nltk
            from nameparser.parser import HumanName
            from nltk.corpus import wordnet


            async def convert_chunks_to_audio(chunks, output_folder):
                audio_files = []  # List to store the paths of generated audio files

                # Iterate over each chunk of text
                for i1, chunk in enumerate(chunks):
                    person_list= []
                    person_names=person_list
                    def get_human_names(text):
                        tokens = nltk.tokenize.word_tokenize(text)
                        pos = nltk.pos_tag(tokens)
                        sentt = nltk.ne_chunk(pos, binary = False)
                        person = []
                        name = ""
                        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
                            for leaf in subtree.leaves():
                                person.append(leaf[0])
                            if len(person) > 1: #avoid grabbing lone surnames
                                for part in person:
                                    name += part + ' '
                                if name[:-1] not in person_list:
                                    person_list.append(name[:-1])
                                name = ''
                            person = []
                    #     print (person_list)


                    get_human_names(chunk)
                            

                    for person in person_list:
                        person_split = person.split(" ")
                        # for name in person_split:
                        #     if wordnet.synsets(name):
                        #         if(name in person):
                        #             person_names.remove(person)
                        #             break
                    print(person_names)

                    print("/-/"*100)
                            
                    g_list = [] 
                    def gender_features(word): 
                        return {'last_letter':word[-1]} 

                    # preparing a list of examples and corresponding class labels. 
                    labeled_names = ([(name, 'male') for name in names.words('male.txt')]+
                                [(name, 'female') for name in names.words('female.txt')]) 

                    random.shuffle(labeled_names) 

                    # we use the feature extractor to process the names data. 
                    featuresets = [(gender_features(n), gender) 
                                for (n, gender)in labeled_names] 

                    # Divide the resulting list of feature 
                    # sets into a training set and a test set. 
                    train_set, test_set = featuresets[500:], featuresets[:500] 

                    # The training set is used to 
                    # train a new "naive Bayes" classifier. 
                    classifier = nltk.NaiveBayesClassifier.train(train_set) 

                    
                    c1,c2 = 0,0
                    g_list = []
                    person_names = [person_names]
                    for i in person_names:
                        for j in i:
                            if classifier.classify(gender_features(j))=='male':
                                c1+=1
                            else:
                                c2+=1
                        if c1>c2:
                            g_list.append('male')
                        elif c1==c2:
                            g_list.append("_")
                        else:
                            g_list.append("female")
                            
                        
                        
                        print("g_list: ",g_list)
                        if g_list[-1]=="_":
                            
                            # Define the path for the output audio file
                            output_file = os.path.join(output_folder, f"part_{i1+1}.mp3")

                            # Convert the text chunk to speech and save as an audio file
                            text_to_speech_female_1(chunk, output_file)
                            await websocket.send_text(f"part_{i1+1}")
                            # Append the path of the created audio file to the list
                            audio_files.append(output_file)
                            
                        elif g_list[-1]=='male':
                            # Define the path for the output audio file
                            output_file = os.path.join(output_folder, f"part_{i1+1}.mp3")

                            # Convert the text chunk to speech and save as an audio file
                            text_to_speech_male_2(chunk, output_file)
                            await websocket.send_text(f"part_{i1+1}")
                            # Append the path of the created audio file to the list
                            audio_files.append(output_file)
                        
                        
                        elif g_list[-1]=='female':
                            # Define the path for the output audio file
                            output_file = os.path.join(output_folder, f"part_{i1+1}.mp3")

                            # Convert the text chunk to speech and save as an audio file
                            text_to_speech_female_2(chunk, output_file)
                            await websocket.send_text(f"part_{i1+1}")
                            # Append the path of the created audio file to the list
                            audio_files.append(output_file)
                            
                        
                        
                            
                        

                return audio_files  # Return the list of audio file paths

            output_folder = "chunks1"  # Define the folder to save audio chunks
            audio_files = await convert_chunks_to_audio(chunks, output_folder)  # Convert chunks to audio files
            print(audio_files) # print list of all the audio files generated

            import re
            from moviepy.editor import concatenate_audioclips, AudioFileClip
            import os

            def extract_number(filename):
                """ Extracts the number from the filename """
                numbers = re.findall(r'\d+', filename)
                return int(numbers[0]) if numbers else 0

            def combine_audio_with_moviepy(folder_path, output_file):
                audio_clips = []  # List to store the audio clips

                # Retrieve and sort files based on the numeric part of the filename
                sorted_files = sorted(os.listdir(folder_path), key=extract_number)

                # Iterate through each sorted file in the given folder
                for file_name in sorted_files:
                    if file_name.endswith('.mp3'):
                        # Construct the full path of the audio file
                        file_path = os.path.join(folder_path, file_name)
                        print(f"Processing file: {file_path}")

                        try:
                            # Create an AudioFileClip object for each audio file
                            clip = AudioFileClip(file_path)
                            audio_clips.append(clip)  # Add the clip to the list
                        except Exception as e:
                            # Print any errors encountered while processing the file
                            print(f"Error processing file {file_path}: {e}")

                # Check if there are any audio clips to combine
                if audio_clips:
                    # Concatenate all the audio clips into a single clip
                    final_clip = concatenate_audioclips(audio_clips)
                    # Write the combined clip to the specified output file
                    final_clip.write_audiofile(output_file)
                    print(f"Combined audio saved to {output_file}")
                else:
                    print("No audio clips to combine.")

            combine_audio_with_moviepy('chunks1', 'chunks1/combined_audio.mp3')  # Combine audio files in 'chunks' folder




        data = await websocket.receive_text()
        if data:
            print("hello world")
            await _main(data)
            await websocket.send_text("combined_audio")
            await websocket.send_text("none")