# In-Course Assistant for Skilljar + Rise

**AI-powered Q&A for your courses**

A floating chat widget that gives learners instant answers to their questions—right inside your Skilljar courses. Built for teams using **Rise** to create content and **Skilljar** to deliver it.

Upload your course content (transcripts, PDFs, documents) through the admin dashboard to train the assistant on your material. When learners get stuck or want clarification, they click the chat bubble and ask. The AI responds using your course content, keeping learners engaged and moving forward.

**Works with the top LLMs**—choose between Claude (Anthropic), GPT (OpenAI), or Gemini (Google) for each course.

**No iframes required.** Unlike most chat widgets, this one doesn't rely on iframes. Add a single script to Skilljar once, and the widget automatically appears on any course you configure in the admin dashboard.

---

## What It Does

- **Answers learner questions** using your actual course content (transcripts, PDFs, documents)
- **Appears automatically** on any Skilljar course you configure
- **Streams responses** in real-time so learners don't wait
- **Suggests questions** to help learners get started
- **Stays on topic** by grounding answers in your course material
- **Works with multiple AI providers** (Claude, GPT, or Gemini—you choose)

---

## What Learners Experience

When a learner opens your course in Skilljar:

1. A small chat bubble appears in the corner of the screen
2. They click it to open the assistant
3. They see suggested questions based on your course content, or type their own
4. The AI answers using your course material
5. They can ask follow-up questions—the assistant remembers the conversation

The assistant is designed to:
- Answer questions using your course material as the primary source
- Clarify concepts with helpful examples
- Admit when something isn't covered in the course
- Gently redirect off-topic questions back to the course

---

## Getting Started

### What You Need

1. **This tool running on a server** (your IT team handles this—see the IT Setup section below)
2. **Your course content** (transcripts, PDFs, Word docs, or text files)
3. **An API key** from at least one AI provider:
   - [Anthropic (Claude)](https://console.anthropic.com/settings/keys) — Recommended
   - [OpenAI (GPT)](https://platform.openai.com/api-keys)
   - [Google (Gemini)](https://aistudio.google.com/apikey)

### Step 1: Open the Admin Dashboard

Your IT team will give you a URL like `https://assistant.yourcompany.com`. Open it in your browser.

On your first visit, you'll create an admin account with your email and password. This first account becomes the administrator who can manage other users.

### Step 2: Add Your API Key

Before creating assistants, you need to connect at least one AI provider:

1. Click **"Settings"** in the top navigation (the gear icon)
2. Select **"API & Privacy"**
3. Paste your API key for the provider(s) you want to use
4. Click **"Save Settings"**

You can add keys for multiple providers. Each course can use a different one.

### Step 3: Create a Course Assistant

1. Click **"+ Create Course Bot"**

2. Fill in the details:
   - **Course Name**: What you'll call it (e.g., "Chrome Fundamentals")
   - **Description**: Optional note for yourself
   - **Course Content**: Upload your transcript, PDF, or Word doc—or paste text directly
   - **AI Model**: Pick which AI powers this assistant
   - **Skilljar Course ID**: The ID that links this to your Skilljar course (see Step 5)

3. Click **"Save Course Bot"**

### Step 4: Add the Script to Skilljar (One Time)

This step connects Skilljar to your assistant server. You only do this once.

1. In the admin dashboard, click **"Add to Skilljar"** (the teal button)
2. Copy the script that appears
3. In Skilljar, go to **Settings → Themes → Global Head Snippet**
4. Paste the script and save

Now the assistant will automatically appear on any course where you've configured a matching Course ID.

### Step 5: Get Your Skilljar Course ID

For each course you want the assistant on:

1. Open your course as a learner would see it
2. Press **F12** to open Developer Tools
3. Click the **Console** tab
4. Type `skilljarCourse.id` and press Enter
5. Copy the value (something like `x24tgec3bed2`)

Then go back to your course bot, click Edit, paste the ID in the Skilljar Course ID field, and save.

**That's it!** The chat bubble will now appear on that course.

---

## Course Scoping (Skilljar Course ID)

The header snippet loads on every page in your Skilljar instance. The widget only renders when the current page belongs to a course you've configured in the app. This keeps installation simple (one global snippet) while still giving you per-course control.

### How Course Scoping Works

1. The Skilljar header script runs on page load.
2. The script reads the current Skilljar course ID from the page.
3. The script checks your app's configured courses.
4. If a matching course bot exists, the script injects the widget. If not, it does nothing.

### Find the Skilljar Course ID

1. Open any page inside the course in Skilljar.
2. Open your browser developer tools:
   - **Chrome / Edge (Windows):** F12
   - **Chrome / Edge (Mac):** Option + Command + I
   - Or right-click the page and select **Inspect**
3. Go to the **Console** tab.
4. Type `skilljarCourse.id` and press Enter.
5. Copy the course ID value (e.g., `x24tgec3bed2`).

**Tip:** If you don't see `skilljarCourse` immediately, refresh the page and try again after the page fully loads.

### Add the Course ID to Your Course Bot

In the admin dashboard, create or edit a course bot and paste the Skilljar Course ID in the designated field. Once saved, reload the course page in Skilljar and confirm the widget appears.

### Scoping Troubleshooting

- **The widget shows up on the wrong course.** Double-check you copied the correct course ID and that you're not reusing an ID from a different environment (staging vs production).
- **I can't find `skilljarCourse`.** Make sure you're on a page inside a course (not the catalog or dashboard), refresh the page, then search the console again.
- **The widget doesn't appear after adding the ID.** Hard refresh the page (Shift + Reload) and confirm the header script is installed in your Skilljar Global Head Snippet.

---

## Multi-User Access

The admin dashboard supports multiple users, so your team can collaborate on managing course assistants.

### User Roles

- **Administrator** (first user created): Can add/remove users, plus all standard permissions
- **Users**: Can create, edit, and delete course bots, manage API keys, view analytics, and customize widget appearance

### Adding Team Members

1. Click your **avatar** in the top-right corner
2. Click **"Settings"** → **"Manage Users"** (admin only)
3. Enter their email and a temporary password
4. Share the credentials—they can change their password after logging in

### Changing Your Password

1. Click your **avatar** in the top-right corner
2. Select **"Account Settings"**
3. Enter your current password and new password
4. Click **"Update Password"**

---

## Customizing the Appearance

Click **"Settings"** → **"Widget Style"** to customize how the chat looks:

- **Theme**: Light or dark mode
- **Colors**: Match your brand colors
- **Position**: Bottom-right or bottom-left corner
- **Text**: Change the title, subtitle, placeholder, and welcome message

Changes apply to all your course assistants.

---

## Tips for Great Results

### What to Include in Your Course Content

The AI can only answer questions about what you give it. Include:

- **Full transcripts** from your Rise course (best results)
- **Key concepts and definitions**
- **Learning objectives**
- **Step-by-step procedures**
- **FAQs** you've already created

### Best Practices

- **More content = better answers** — The AI works with what you provide
- **Include context** — "In Module 3, we cover..." helps the AI give specific answers
- **Update when your course changes** — Keep the content in sync
- **Use clear headings** — Helps the AI find relevant information

### Supported File Formats

- **PDF** — Great for exported documents
- **Word (.docx)** — Excellent text extraction
- **Text (.txt)** — Perfect for transcripts
- **Markdown (.md)** — Good for structured content

---

## Managing Your Course Assistants

### Edit an Assistant
Click **Edit** on any course card, make changes, and save. Changes take effect immediately.

### Delete an Assistant
Click **Delete** on the course card. This removes it permanently and the chat bubble will stop appearing on that course.

### Preview
Open the Skilljar course to see exactly what learners experience.

---

## Content Intelligence (Analytics)

Click **"Analytics"** in the navigation to see insights about how learners use your course assistants.

### What You Can See

- **Overview Stats**: Total conversations, content gaps detected, grounding percentage
- **Content Gaps**: Questions the AI couldn't answer from your course content—these reveal what's missing
- **Lesson Friction**: Which lessons generate the most questions (heatmap view)
- **Conversations**: Review actual chat sessions (requires Full Logging enabled)

### How It Works

The AI automatically tags each response with metadata:
- **Grounding source**: Was the answer from course content, general knowledge, or uncertain?
- **Content gap detected**: Did the learner ask about something not covered?
- **Gap topic**: What topic was missing?

This helps you identify where to improve your course content.

### Privacy Controls

You control how much data is stored. Go to **Settings** → **API & Privacy**:

- **Analytics Only** (recommended): Tracks patterns without storing message content
- **Full Logging**: Stores complete conversations for review
- **Disabled**: No analytics collected

---

## Troubleshooting

### The chat bubble doesn't appear on my course
- Double-check that the Skilljar Course ID matches exactly
- Make sure the global script was added to Skilljar's Global Head Snippet
- Check that your server is running (ask IT)

### The AI gives wrong or incomplete answers
- Check that your course content is complete
- Make sure transcripts are accurate
- Update the content if your course has changed

### The widget looks wrong
- Click "Widget Style" and check your settings
- Changes appear on page refresh

---

---

## For IT/DevOps: Server Setup

This section is for the technical team deploying the server.

### Requirements

- Python 3.9+
- API keys are configured via the web UI after deployment

### Quick Start

```bash
cd in-course-assistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt

cd backend
python main.py
```

Server runs at `http://localhost:8000`. On first visit, the setup wizard creates an admin account.

### Environment Variables

All configuration happens via the web UI. Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated allowed CORS origins (set for production) |
| `RATE_LIMIT_PER_MINUTE` | `20` | Max requests per IP per minute |
| `MAX_CONTENT_LENGTH` | `500000` | Max course content characters |

### Production Deployment

#### Render.com (Recommended)

1. Create a new **Web Service**
2. Connect your repository
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `cd backend && python main.py`
5. Add `ALLOWED_ORIGINS` environment variable
6. Deploy

#### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "backend/main.py"]
```

```bash
docker build -t in-course-assistant .
docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://yourdomain.com in-course-assistant
```

### Security Checklist

- [ ] Use HTTPS in production
- [ ] Set `ALLOWED_ORIGINS` to specific domains
- [ ] Use a strong admin password
- [ ] Set up billing alerts on AI provider accounts
- [ ] Rotate API keys periodically

### Architecture

```
in-course-assistant/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── database.py       # Data models
│   └── requirements.txt
├── frontend/
│   ├── admin.html        # Admin dashboard
│   └── setup.html        # First-run wizard
├── widget/
│   ├── widget.html       # Chat widget (iframe)
│   └── widget-loader.js  # Skilljar integration script
└── README.md
```

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/` | Admin dashboard |
| `/setup` | First-run setup |
| `/widget.html?course_id=XXX` | Embedded chat widget |
| `/widget-loader.js` | Skilljar loader script |
| `/health` | Health check |
| `/api/courses` | Course CRUD |
| `/api/chat/stream` | Streaming chat endpoint |
| `/api/settings` | API key and privacy settings |
| `/api/widget-style` | Widget appearance settings |
| `/api/users` | User management (admin only) |
| `/api/users/me` | Current user info and password change |
| `/api/analytics/*` | Content intelligence analytics |

### Data Storage

| Data | Stored | Notes |
|------|--------|-------|
| User credentials | Yes | Hashed with bcrypt, multiple users supported |
| API keys | Yes | Per-provider storage |
| Course content | Yes | Text content + settings |
| Chat analytics | Configurable | See Privacy Settings below |
| Learner identity | No | No PII collected |

### Privacy Settings

The admin can control what chat data is stored for analytics:

| Setting | What's Stored | Analytics Available |
|---------|---------------|---------------------|
| **Analytics Only** (default) | Metadata only (grounding source, gaps, response times) | Content gaps, lesson friction, question clustering |
| **Full Logging** | Complete conversation history | All analytics + conversation review |
| **Disabled** | Nothing | No analytics |

Configure this in **Settings** → **API & Privacy** → **Analytics & Conversation Logging**.

### Third-Party AI Services

This tool sends course content and learner questions to third-party AI providers (Anthropic, OpenAI, or Google) via their APIs. You are responsible for:

- Reviewing each provider's terms of service and data policies
- Ensuring compliance with your organization's data governance requirements
- Obtaining any necessary approvals before uploading proprietary content

As of January 2026, these providers state that API usage is not used for model training, but policies may change. This project makes no guarantees about third-party data handling.

---

## Credits

Created by **Kelly Mullaney** ([kelly.r.mullaney@gmail.com](mailto:kelly.r.mullaney@gmail.com))

---

## License

MIT License — free to use and modify.
