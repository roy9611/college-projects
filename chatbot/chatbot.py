import nltk
from nltk.chat.util import Chat, reflections

# Ensure that the necessary resources are downloaded
nltk.download('punkt')

# Define pairs of input and output
pairs = [
    (r"Hi|Hello|Hey", ["Hello!", "Hi there!", "Hey!"]),
    (r"How are you?", ["I'm doing great, thanks for asking!", "I'm good, how about you?"]),
    (r"What's your name?", ["I am a chatbot, I don't have a name!"]),
    (r"(.*) your favorite (.*)", ["I love talking to people about various things!"]),
    (r"(.*) help(.*)", ["Sure! How can I assist you?"]),
    (r"Quit", ["Goodbye!", "See you later!"]),
    (r"How old are you?", ["I don't age. I am forever young!"]),
    (r"(.*) feel(.*)", ["I'm just a chatbot, I don't have feelings, but I am always here to chat!"]),
    (r"Tell me a joke", ["Why don't skeletons fight each other? They don't have the guts!"]),
    (r"(.*) name(.*)", ["My name is ChatBot, nice to meet you!"]),
    (r"Who are you?", ["I am just a friendly chatbot here to help you with anything you need."]),
    (r"Where do you live?", ["I live in the cloud, always online!"]),
    (r"How can I contact you?", ["You can talk to me right here!"]),
    (r"(.*) your favorite color(.*)", ["I don't have a favorite color, but I like all the colors!"]),
    (r"(.*) movie(.*)", ["I enjoy many movies, but I don't have preferences. What about you?"]),
    (r"(.*) music(.*)", ["I love all kinds of music. What kind of music do you like?"]),
    (r"Tell me something interesting", ["Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs!"]),
    (r"Tell me a story", ["Once upon a time, in a land full of code and algorithms, a chatbot lived to help people like you!"]),
    (r"(.*) code(.*)", ["Are you a programmer? I can help with coding!"]),
    (r"(.*) python(.*)", ["Python is an awesome programming language! What do you like to do with Python?"]),
    (r"(.*) math(.*)", ["Math is fun! Are you working on a math problem?"]),
    (r"(.*) weather(.*)", ["I don't have access to weather updates, but you can check your local forecast online."]),
    (r"Quit", ["Goodbye! Have a great day!"]),
    (r"(.*) hobbies(.*)", ["I enjoy chatting with people! What are your hobbies?"]),
    (r"(.*) favorite food(.*)", ["I don't eat, but I hear pizza is quite popular!"]),
    (r"(.*) sports(.*)", ["I think sports are exciting! Do you have a favorite sport?"]),
    (r"(.*) travel(.*)", ["Traveling is a great way to explore the world! Where would you like to go?"]),
    (r"(.*) books(.*)", ["I love the idea of books! What genre do you enjoy?"]),
    (r"(.*) technology(.*)", ["Technology is fascinating! What tech are you interested in?"]),
    (r"(.*) news(.*)", ["I don't have access to current news, but you can check online!"]),
    (r"(.*) health(.*)", ["Health is important! Do you have any health tips?"]),
    (r"(.*) advice(.*)", ["I'm here to listen! What do you need advice on?"]),
    (r"(.*) life(.*)", ["Life is a journey! What are your thoughts on it?"]),
    (r"(.*) future(.*)", ["The future is full of possibilities! What do you hope for?"]),
    (r"(.*) dreams(.*)", ["Dreams can be inspiring! What do you dream about?"]),
    (r"(.*) favorite season(.*)", ["I don't experience seasons, but I hear spring is lovely!"]),
    (r"(.*) pets(.*)", ["Pets are wonderful companions! Do you have any?"]),
    (r"(.*) family(.*)", ["Family is important! How is your family?"]),
    (r"(.*) friends(.*)", ["Friends make life better! Who is your best friend?"]),
    (r"(.*) food(.*)", ["Food brings people together! What's your favorite dish?"]),
    (r"(.*) art(.*)", ["Art is a beautiful form of expression! Do you create art?"]),
    (r"(.*) dance(.*)", ["Dancing is a fun way to express yourself! Do you like to dance?"]),
    (r"(.*) sing(.*)", ["Singing can be so joyful! Do you enjoy singing?"]),
    (r"(.*) nature(.*)", ["Nature is amazing! Do you have a favorite place in nature?"]),
    (r"(.*) science(.*)", ["Science helps us understand the world! What area of science interests you?"]),
    (r"(.*) history(.*)", ["History teaches us valuable lessons! Do you have a favorite historical figure?"]),
    (r"(.*) philosophy(.*)", ["Philosophy encourages deep thinking! What philosophical questions do you ponder?"]),
    (r"(.*) travel(.*)", ["Traveling opens up new perspectives! Where have you traveled?"]),
    (r"(.*) culture(.*)", ["Culture enriches our lives! What aspects of culture do you enjoy?"]),
    (r"(.*) environment(.*)", ["The environment is crucial! What do you do to help the planet?"]),
    (r"(.*) climate change(.*)", ["Climate change is a pressing issue! What are your thoughts on it?"]),
    (r"(.*) sustainability(.*)", ["Sustainability is important for our future! How do you practice it?"]),
    (r"(.*) goals(.*)", ["Setting goals is essential! What are your current goals?"]),
    (r"(.*) motivation(.*)", ["Motivation drives us forward! What motivates you?"]),
    (r"(.*) success(.*)", ["Success means different things to different people! What does it mean to you?"]),
    (r"(.*) failure(.*)", ["Failure is a part of learning! How do you handle setbacks?"]),
    (r"(.*) happiness(.*)", ["Happiness is key to a fulfilling life! What makes you happy?"]),
    (r"(.*) sadness(.*)", ["It's okay to feel sad sometimes! How do you cope with sadness?"]),
    (r"(.*) stress(.*)", ["Stress management is important! What do you do to relax?"]),
    (r"(.*) mindfulness(.*)", ["Mindfulness can help with stress! Do you practice it?"]),
    (r"(.*) meditation(.*)", ["Meditation can be calming! Have you tried it?"]),
    (r"(.*) exercise(.*)", ["Exercise is great for health! What type of exercise do you enjoy?"]),
    (r"(.*) diet(.*)", ["A balanced diet is important! What do you like to eat?"]),
    (r"(.*) cooking(.*)", ["Cooking can be fun! Do you enjoy making meals?"]),
    (r"(.*) baking(.*)", ["Baking is a delightful activity! What do you like to bake?"]),
    (r"(.*) gardening(.*)", ["Gardening can be therapeutic! Do you have a garden?"]),
    (r"(.*) hobbies(.*)", ["Hobbies enrich our lives! What are your favorite hobbies?"]),
    (r"(.*) learning(.*)", ["Learning is a lifelong journey! What are you currently learning?"]),
    (r"(.*) skills(.*)", ["Skills can be developed over time! What skills are you working on?"]),
    (r"(.*) challenges(.*)", ["Challenges help us grow! What challenges have you faced?"]),
    (r"(.*) achievements(.*)", ["Achievements are worth celebrating! What are you proud of?"]),
    (r"(.*) dreams(.*)", ["Dreams can inspire us! What are your dreams?"]),
    (r"(.*) aspirations(.*)", ["Aspirations guide our paths! What are your aspirations?"]),
    (r"(.*) future(.*)", ["The future is full of possibilities! What do you hope for?"]),
    (r"(.*)", ["Sorry, I didn't understand that. Could you rephrase?"]),
]

# Create chatbot instance
chatbot = Chat(pairs, reflections)

# Function to handle chatbot responses
def chatbot_response():
    print("Start chatting with the bot (type 'Quit' to exit):")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Bot: Goodbye!")
                break
            response = chatbot.respond(user_input)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break

# Extended functionality
def extended_conversation():
    print("Extended Conversation Mode Activated! You can type 'Quit' to exit anytime.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Bot: Goodbye!")
                break
            elif 'hello' in user_input.lower():
                print("Bot: Hi there, how can I help you today?")
            elif 'love' in user_input.lower():
                print("Bot: Love is a beautiful thing, don't you think?")
            elif 'weather' in user_input.lower():
                print("Bot: I can't check the weather, but it's always sunny where I live!")
            else:
                response = chatbot.respond(user_input)
                print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break

# Function to guide users through available commands
def display_help():
    print("""
    Available Commands:
    1. Type 'Hello' or 'Hi' to start the conversation.
    2. Ask about my favorite things, such as color, movie, or music.
    3. Ask for a joke or fun facts.
    4. Type 'Quit' to exit the chat.
    5. You can also talk to me about coding, math, and many other things!
    6. Type 'Extended' for a more interactive conversation.
    7. Ask about hobbies, interests, and personal development.
    8. Inquire about health, wellness, and lifestyle tips.
    9. Discuss current events, technology, and science.
    10. Share your thoughts on philosophy, culture, and the environment.
    """)

# Main function to choose between different modes
def main():
    print("Welcome to the Chatbot!")
    print("Type 'Help' to see available commands.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Bot: Goodbye!")
                break
            elif user_input.lower() == 'help':
                display_help()
            elif 'start' in user_input.lower():
                chatbot_response()
            elif 'extended' in user_input.lower():
                extended_conversation()
            else:
                print("Bot: I'm here to chat! Type 'Help' for available commands.")
        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break

if __name__ == "__main__":
    main()