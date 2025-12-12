# 🔑 How to Get Your Free Hugging Face Token

Follow these simple steps to get your free Hugging Face API token and improve your chatbot's performance:

## Step 1: Create a Free Hugging Face Account

1. Go to **https://huggingface.co/join**
2. Sign up with your email (it's completely FREE!)
3. Verify your email address

## Step 2: Get Your API Token

1. Once logged in, go to **https://huggingface.co/settings/tokens**
2. Click on **"New token"** button
3. Give it a name (e.g., "PlantDocBot")
4. Select **"Read"** access (this is all you need)
5. Click **"Generate token"**
6. **Copy the token** (it will look like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxx`)

## Step 3: Add Token to Your Project

1. Open the file: `f:\project\Resume_CV_Project\Plant_chat_bot\Backend\.env`
2. Replace `your_token_here` with your actual token:
   ```
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Save the file

## Step 4: Restart the Backend Server

The backend server will automatically reload and pick up the new token!

## ✅ Benefits of Adding a Token

- ⚡ **Faster responses** - Priority access to the API
- 🚀 **Higher rate limits** - More requests per hour
- 💯 **Better reliability** - Less likely to hit rate limits
- 🆓 **Still completely FREE!**

## 🔒 Security Note

- The `.env` file is already added to `.gitignore` (if you have one)
- Never share your token publicly
- Never commit the `.env` file to GitHub

## 📝 Current Status

✅ `.env` file created at: `Backend/.env`
✅ `python-dotenv` installed
✅ Backend configured to use the token
⏳ **Next step**: Add your token to the `.env` file

## 🧪 Testing

After adding your token:
1. The backend will automatically reload
2. Open http://localhost:5173
3. Try the chatbot - it should respond faster!

---

**Note**: The chatbot works WITHOUT a token too, but adding one gives you better performance!
