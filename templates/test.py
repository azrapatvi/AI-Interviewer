import google.generativeai as genai

# Configure your API key
genai.configure(api_key="AIzaSyCG2-HhRq04PxVri_BRZusB4gvqyRvCQzQ")

# Choose an available model
model = genai.GenerativeModel("gemini-2.5-flash")

# Generate a simple response
response = model.generate_content("Say hello")

# Print the model output
print(response.text)
