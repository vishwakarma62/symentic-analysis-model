import streamlit as st

st.title("🚀 DEPLOYMENT TEST")
st.success("✅ If you see this, your deployment is working!")
st.write("Current timestamp:", st.empty())

st.header("Simple Sentiment Test")
text = st.text_input("Enter text:")
if text:
    if any(word in text.lower() for word in ['good', 'great', 'love', 'excellent']):
        st.success("😊 POSITIVE")
    elif any(word in text.lower() for word in ['bad', 'terrible', 'hate', 'awful']):
        st.error("😞 NEGATIVE") 
    else:
        st.info("😐 NEUTRAL")

st.write("This is a minimal test app to verify Railway deployment works.")