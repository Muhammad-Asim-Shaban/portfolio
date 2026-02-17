import { IProject } from '@/types';

export const GENERAL_INFO = {
    email: 'asimshaban43@gmail.com',

    emailSubject: "Let's collaborate on a project",
    emailBody: 'Hi Asim, I am reaching out to you because...',

    // oldPortfolio: 'https://www.legacy.me.toinfinite.dev',
    // upworkProfile: 'https://www.upwork.com/freelancers/tajmirul',
};

export const SOCIAL_LINKS = [
    { name: 'github', url: 'https://github.com/Muhammad-Asim-Shaban' },
    { name: 'linkedin', url: 'https://www.linkedin.com/in/muhammad-asim-shaban-00a731294/' },
    { name: 'facebook', url: 'https://www.facebook.com/' },
    // { name: 'Old Version', url: GENERAL_INFO.oldPortfolio },
];

export const MY_STACK = {
    llm_and_agents: [
        {
            name: 'LangChain',
            icon: '/logo/langchain.svg',
        },
        {
            name: 'Hugging Face',
            icon: '/logo/huggingface.svg',
        },
        {
            name: 'Ollama',
            icon: '/logo/ollama.svg',
        },
        {
            name: 'n8n',
            icon: '/logo/n8n.svg',
        },
       
    ],
    computer_vision:[
        {
            name: 'Yolov11',
            icon: "/logo/yolo.svg"
        },
        {
            name: "OpenCV",
            icon:"/logo/opencv.svg"
        },
        {
            name: "MediaPipe",
            icon:"/logo/mediapipe.svg"
        }
    ],
    ML_AND_Deep_learning:[
        {
            name: "scikit learn",
            icon: "/logo/scikit-learn.svg"
        },
        {
            name: "Tensor Flow",
            icon: "/logo/TensorFlow.svg"
        },  
    ],
    backend: [
        {
            name: 'Node.js',
            icon: '/logo/node.png',
        },
        {
            name: 'Express.js',
            icon: '/logo/express.png',
        },
        {
            name: 'Php',
            icon: '/logo/php.svg'
        }
    ],
    database: [
        {
            name: 'MySQL',
            icon: '/logo/mysql.svg',
        },
        {
            name: 'PostgreSQL',
            icon: '/logo/postgreSQL.png',
        },
        {
            name: 'MongoDB',
            icon: '/logo/mongodb.svg',
        },
    ],
    tools: [
        {
            name: 'Git',
            icon: '/logo/git.png',
        },
        {
            name: 'Docker',
            icon: '/logo/docker.svg',
        },
    ],
};


export const PROJECTS: IProject[] = [
    
    {
    title: 'Personal Assistant Agent (n8n + Ollama)',
    slug: 'personal-assistant-agent-n8n',
    // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/n8n-personal-assistant-agent',
    year: 2025,
    description: `
  An autonomous Personal Assistant Agent built with n8n workflow automation and powered by the GLM-5 language model running locally through Ollama. The agent connects to the entire Google ecosystem — Calendar, Gmail, Tasks, Docs, and Sheets — to automate real daily productivity tasks without relying on cloud-hosted LLMs.<br/><br/>

  Key Features:<br/>
  <ul>
    <li>🗓️ Calendar Management: Schedules meetings and events directly via Google Calendar</li>
    <li>📧 Email Automation: Reads, summarizes, and sends replies through Gmail autonomously</li>
    <li>✅ Task Management: Creates and manages to-do lists with Google Tasks</li>
    <li>📝 Smart Notes: Takes and organizes quick notes directly in Google Docs</li>
    <li>💰 Expense Tracking: Logs and tracks budgeting data into Google Sheets</li>
    <li>💬 Q&A Interface: Answers general knowledge questions powered by GLM-5</li>
  </ul><br/>

  Technical Highlights:
  <ul>
    <li>Built entirely as a visual n8n workflow — exported as reusable JSON for easy import</li>
    <li>Runs GLM-5 100% locally via Ollama — no API keys or cloud costs required</li>
    <li>Integrated 5 Google APIs (Calendar, Gmail, Tasks, Docs, Sheets) in a single agent pipeline</li>
    <li>Designed for extensibility — new tools and automations can be added as n8n nodes</li>
  </ul>
    `,
    role: `
  Solo AI Automation Engineer — designed and built the entire agent:<br/>
  <ul>
    <li>🔧 Workflow Design: Architected the full n8n agent workflow with tool-routing logic</li>
    <li>🤖 LLM Integration: Connected GLM-5 via Ollama as the local reasoning backbone</li>
    <li>🔌 Google API Setup: Configured OAuth credentials and integrated all 5 Google services</li>
    <li>🐍 Custom Logic: Wrote Python helper scripts for additional processing tasks</li>
    <li>📦 Packaging: Exported workflow as portable JSON with full setup documentation</li>
  </ul>
    `,
    techStack: [
        'n8n',
        'Ollama',
        'GLM-5',
        'Google Calendar API',
        'Gmail API',
        'Google Tasks API',
        'Google Docs API',
        'Google Sheets API',
        'Python',
    ],
    thumbnail: '/projects/thumbnail/11.png',
        longThumbnail: '/projects/long/12.png',
        images: [
        '/projects/images/11.png',
        '/projects/images/12.png',
    ],
},
    {
        title: 'Multi-Agent Travel Planner',
        slug: 'travel-planner-multiagent',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Travel-Planner---MultiAI-Agent',
        year: 2025,
        description: `
      An intelligent travel planning system powered by a collaborative network of specialized AI agents. Each agent handles a distinct part of the trip-planning pipeline — from destination research to itinerary generation — and they communicate to produce a complete, personalized travel plan. <br/><br/>

      Key Features:<br/>
      <ul>
        <li>🤖 Multi-Agent Architecture: Specialized agents for destination analysis, local expertise, and itinerary creation</li>
        <li>🗺️ End-to-End Planning: Handles destination research, weather, budget estimation, and daily itinerary generation</li>
        <li>🔗 Agent Orchestration: Agents collaborate in a supervisor-worker pattern with context passing between steps</li>
        <li>💬 Natural Language Interface: Accepts free-form travel requests and returns structured, readable plans</li>
        <li>📋 Structured Output: Produces day-by-day itineraries with activities, timings, and recommendations</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Implemented LangChain agent executor with custom tool integrations</li>
        <li>Designed prompt templates for each specialized agent role</li>
        <li>Built inter-agent communication pipeline with shared state management</li>
        <li>Used OpenAI GPT models as the reasoning backbone for each agent</li>
      </ul>
        `,
        role: `
      Solo AI Engineer — designed and built the entire system:<br/>
      <ul>
        <li>🧠 Agent Design: Defined agent roles, responsibilities, and tool access for each node</li>
        <li>🔗 Orchestration: Built the multi-agent pipeline with LangChain and prompt chaining</li>
        <li>🛠️ Tool Integration: Connected agents to external tools for data retrieval and reasoning</li>
        <li>🧪 Testing: Evaluated agent outputs across diverse travel scenarios</li>
        <li>📦 Packaging: Structured the project for reproducibility with requirements and clear README</li>
      </ul>
        `,
        techStack: [
            'Python',
            'LangChain',
            'Multi-Agent AI',
            'Prompt Engineering',
        ],
        thumbnail: '',
        longThumbnail: '',
        images: [
        '',
        '',
    ],
    },
    {
    title: 'Malicious File Detector',
    slug: 'malicious-file-detector',
    // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Malicious-File-Detector',
    year: 2025,
    description: `
  A full-stack cybersecurity application available on both Web and Mobile that analyzes uploaded files for malicious behavior using three independent detection engines. Each engine runs in parallel and the combined results are presented to the user in a clear, detailed report — giving both a verdict and the reasoning behind it.<br/><br/>

  Key Features:<br/>
  <ul>
    <li>🔏 Signature-Based Detection: Matches file hashes and byte patterns against a known malware signature database</li>
    <li>🧠 Heuristics-Based Detection: Analyzes file structure, metadata, and behavioral indicators for suspicious patterns without needing a known signature</li>
    <li>🐳 Sandbox Execution: Runs the file inside an isolated Docker container and monitors all system activity — process creations, process deletions, file creations, registry changes, and network calls</li>
    <li>📊 Unified Report: All three detection results are combined into a single structured report shown to the user</li>
    <li>🌐 Web App: Built with React frontend and FastAPI backend for browser-based file analysis</li>
    <li>📱 Mobile App: Built with Flutter frontend and FastAPI backend for on-the-go file scanning</li>
  </ul><br/>

  Technical Highlights:
  <ul>
    <li>Designed a three-engine detection pipeline running signature, heuristics, and sandbox analysis independently</li>
    <li>Sandboxed file execution inside Docker with real-time monitoring of process tree, file system events, and system calls</li>
    <li>Built two separate FastAPI backends — one serving the React web app, one serving the Flutter mobile app</li>
    <li>Implemented file upload, validation, and async processing pipeline in FastAPI</li>
    <li>Flutter mobile app communicates with the FastAPI backend via REST API for cross-platform support</li>
  </ul>
    `,
    role: `
  Solo Full-Stack AI/Security Engineer — built the entire system across web and mobile:<br/>
  <ul>
    <li>🌐 Web Frontend: Developed the React interface for file upload and results display</li>
    <li>📱 Mobile Frontend: Built the Flutter app with file picker and results screen for iOS and Android</li>
    <li>⚙️ Backend (Web): Designed and built the FastAPI backend serving the React web application</li>
    <li>⚙️ Backend (Mobile): Built a separate FastAPI backend optimized for Flutter mobile API calls</li>
    <li>🔏 Signature Engine: Implemented hash-based and byte-pattern matching against malware signature database</li>
    <li>🧠 Heuristics Engine: Developed rule-based analysis of file structure and suspicious behavioral indicators</li>
    <li>🐳 Sandbox Engine: Configured Docker-based isolated execution environment with system call monitoring</li>
    <li>📊 Report System: Aggregated all three engine outputs into a unified, user-readable verdict</li>
  </ul>
    `,
    techStack: [
        'React',
        'Flutter',
        'Python',
        'FastAPI',
        'Docker',
        'Sandbox Analysis',
        'Signature Detection',
        'Heuristics Detection',
        'REST API',
    ],
     thumbnail: '/projects/thumbnail/s31.png',
        longThumbnail: '/projects/long/s32.png',
        images: [
        '/projects/images/s31.png',
        '/projects/images/s32.png',
        '/projects/images/s33.png',
        '/projects/images/s34.png'
    ],
},
    {
        title: 'Document Q&A with LangChain (RAG)',
        slug: 'qa-over-documents-langchain',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Q-A-Over-Documents-Using-Lang-Chain',
        year: 2025,
        description: `
      A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask natural language questions — receiving accurate, context-grounded answers powered by LLMs. The system retrieves only the most relevant document chunks before generating responses, avoiding hallucinations.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>📄 Document Ingestion: Supports PDF and text documents with automatic chunking</li>
        <li>🔍 Semantic Search: Vector embeddings enable similarity-based retrieval over document content</li>
        <li>💡 Context-Aware Answers: LLM generates answers strictly grounded in the retrieved context</li>
        <li>🗂️ Vector Store Integration: Persistent vector database for fast and accurate retrieval</li>
        <li>🔄 Conversation Memory: Maintains chat history for follow-up question handling</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Implemented full RAG pipeline: load → split → embed → store → retrieve → generate</li>
        <li>Used LangChain's document loaders and text splitters for preprocessing</li>
        <li>Integrated OpenAI embeddings with ChromaDB/FAISS for vector storage</li>
        <li>Built conversational retrieval chain with memory buffer</li>
      </ul>
        `,
        role: `
      Solo AI Engineer:<br/>
      <ul>
        <li>📐 Architecture: Designed the end-to-end RAG pipeline from ingestion to response</li>
        <li>🔢 Embeddings: Configured chunking strategy and embedding model selection</li>
        <li>🧠 Chain Design: Built LangChain retrieval chain with custom prompt templates</li>
        <li>🧪 Evaluation: Tested retrieval accuracy and answer quality across document types</li>
      </ul>
        `,
        techStack: [
            'Python',
            'LangChain',
            'OpenAI',
            'ChromaDB / FAISS',
            'RAG Pipeline',
            'Vector Embeddings',
        ],
        thumbnail: '/projects/thumbnail/31.png',
        longThumbnail: '/projects/long/32.png',
        images: [
        '/projects/images/31.png',
        '/projects/images/32.png',
    ],
    },
    {
        title: 'Real-Time People Counter (YOLOv8)',
        slug: 'people-counter-yolov11',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/People-Counter-Using-Yolo-11',
        year: 2025,
        description: `
      A real-time computer vision system that detects and counts people in live video streams using the  YOLOv8 object detection model. The system tracks individual people across frames and maintains accurate entry/exit counts using a virtual line-crossing mechanism.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>👁️ Real-Time Detection: Processes live video at high FPS with YOLOv11 inference</li>
        <li>🔢 Accurate Counting: Virtual line-crossing logic counts people entering and exiting</li>
        <li>🎯 Multi-Object Tracking: Persistent IDs assigned to each tracked individual across frames</li>
        <li>📊 Live Statistics: Real-time count overlay rendered directly on the video feed</li>
        <li>📷 Multi-Source Support: Works with webcam, RTSP streams, and video files</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Deployed YOLOv8 from Ultralytics for state-of-the-art detection accuracy</li>
        <li>Integrated ByteTrack/SORT for robust multi-object tracking persistence</li>
        <li>Implemented directional line-crossing algorithm for in/out counting</li>
        <li>Optimized inference pipeline for minimal latency on CPU and GPU</li>
      </ul>
        `,
        role: `
      Solo CV Engineer:<br/>
      <ul>
        <li>🧠 Model Integration: Set up YOLOv8 inference pipeline with Ultralytics SDK</li>
        <li>🔗 Tracking Logic: Implemented object ID persistence across video frames</li>
        <li>📐 Counting Algorithm: Designed virtual line and directional crossing detection</li>
        <li>🎨 Visualization: Built real-time OpenCV overlay for bounding boxes and stats</li>
      </ul>
        `,
        techStack: [
            'Python',
            'YOLOv8',
            'OpenCV',
            'Ultralytics',
            'Object Tracking',
            'Computer Vision',
        ],
        thumbnail: '/projects/thumbnail/41.png',
        longThumbnail: '/projects/long/41.png',
        images: [
        '/projects/images/41.png',
        '/projects/images/42.png',
    ],
    },
    {
        title: 'AI Caption Generator Web App',
        slug: 'ai-caption-generator',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/AI-Caption-Generator-Web-App',
        year: 2025,
        description: `
      A full-stack AI web application that generates creative, context-aware captions for uploaded images using multimodal AI models. Users upload an image, select a tone (professional, funny, poetic, etc.), and receive multiple caption options instantly.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>🖼️ Image Upload & Preview: Drag-and-drop image upload with instant preview</li>
        <li>✍️ Tone Selection: Multiple caption styles — professional, casual, humorous, poetic</li>
        <li>⚡ Instant Generation: Fast multimodal inference delivers captions in seconds</li>
        <li>📋 One-Click Copy: Copy any generated caption to clipboard immediately</li>
        <li>🌐 Web Interface: Clean, user-friendly UI accessible from any browser</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Integrated multimodal LLM (GPT-4 Vision / Gemini) for image understanding</li>
        <li>Built REST API backend with Python for image processing and AI inference</li>
        <li>Deployed as a complete web application with frontend and backend integration</li>
        <li>Implemented prompt engineering for tone-controlled caption generation</li>
      </ul>
        `,
        role: `
      Full-Stack AI Developer — built the entire product solo:<br/>
      <ul>
        <li>🌐 Frontend: Designed and built the web UI with image upload and caption display</li>
        <li>⚙️ Backend: Developed the API layer handling image encoding and LLM calls</li>
        <li>🧠 AI Integration: Connected multimodal vision model for image captioning</li>
        <li>🚀 Deployment: Packaged and deployed the complete application end-to-end</li>
      </ul>
        `,
        techStack: [
            'Python',
            'GPT-4 Vision',
            'Multimodal AI',
            'REST API',
            'Web App',
            'Prompt Engineering',
        ],
        thumbnail: '',
        longThumbnail: '',
        images: [
        '',
        '',
    ],
    },
    {
        title: 'Text Summarizer (LangChain)',
        slug: 'text-summarizer-langchain',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Text-Summarizer-Using-LangChain',
        year: 2025,
        description: `
      An intelligent text summarization tool built with LangChain that condenses long documents, articles, and reports into concise, coherent summaries. Supports multiple summarization strategies including map-reduce for large documents and refine chains for iterative improvement.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>📄 Long Document Support: Handles documents exceeding LLM context limits via chunking strategies</li>
        <li>🗜️ Multiple Strategies: Stuff, Map-Reduce, and Refine summarization chains</li>
        <li>🎛️ Configurable Length: Control summary verbosity from one-line to detailed</li>
        <li>📋 Bullet or Prose: Choose structured bullet points or flowing paragraph summaries</li>
        <li>🔌 Multi-Format Input: Accepts plain text, PDFs, and web URLs as input sources</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Implemented map-reduce chain to summarize documents beyond context window limits</li>
        <li>Designed custom prompt templates for controlled summarization style and length</li>
        <li>Used LangChain's document loaders for multi-format input handling</li>
        <li>Benchmarked summarization quality across different chain strategies</li>
      </ul>
        `,
        role: `
      Solo AI Engineer:<br/>
      <ul>
        <li>🔗 Chain Design: Built and configured LangChain summarization chains</li>
        <li>📝 Prompt Engineering: Crafted prompts for tone, length, and structure control</li>
        <li>🧪 Evaluation: Compared quality across Stuff, Map-Reduce, and Refine strategies</li>
        <li>📦 Packaging: Documented setup and usage for reproducibility</li>
      </ul>
        `,
        techStack: [
            'Python',
            'LangChain',
            'Prompt Engineering',
        ],
        thumbnail: '/projects/thumbnail/61.png',
        longThumbnail: '/projects/long/62.png',
        images: [
        '/projects/images/61.png',
        '/projects/images/62.png',
    ],
    },
    {
        title: 'Online Fraud Detection',
        slug: 'online-fraud-detection',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Online-Fraud-Detection',
        year: 2025,
        description: `
      A machine learning system that detects fraudulent online financial transactions in real time. Trained on imbalanced real-world transaction data, the model achieves high precision and recall for the minority fraud class — making it practically deployable in fintech pipelines.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>🛡️ Fraud Classification: Binary classifier distinguishing legitimate vs. fraudulent transactions</li>
        <li>⚖️ Imbalance Handling: SMOTE and class-weighting techniques to address severe class imbalance</li>
        <li>📊 Feature Engineering: Transaction amount, frequency, time patterns, and merchant features</li>
        <li>🔍 Explainability: SHAP values reveal which features drive fraud predictions</li>
        <li>📈 Performance Metrics: Optimized for F1-score and AUC-ROC on imbalanced datasets</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Applied XGBoost and Random Forest with hyperparameter tuning via GridSearchCV</li>
        <li>Used SMOTE oversampling to balance training data for better minority class recall</li>
        <li>Generated SHAP feature importance plots for model interpretability</li>
        <li>Evaluated with confusion matrix, precision-recall curve, and ROC-AUC</li>
      </ul>
        `,
        role: `
      Solo ML Engineer — end-to-end model development:<br/>
      <ul>
        <li>📊 Data Analysis: Explored transaction distributions and identified fraud patterns</li>
        <li>⚙️ Preprocessing: Handled missing values, encoded categoricals, scaled features</li>
        <li>🤖 Modeling: Trained and compared multiple classifiers with cross-validation</li>
        <li>🔍 Explainability: Added SHAP analysis for business-facing model interpretability</li>
        <li>📋 Reporting: Documented methodology and results in structured Jupyter Notebook</li>
      </ul>
        `,
        techStack: [
            'Python',
            'XGBoost',
            'scikit-learn',
            'SMOTE',
            'SHAP',
            'Pandas',
            'Matplotlib',
        ],
        thumbnail: '',
        longThumbnail: '',
        images: [
        '',
        '',
    ],
    },
    {
        title: 'Car Counter',
        slug: 'car-counter-yolo11',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Car-Counter-Application-using-Yolo11',
        year: 2025,
        description: `
      A real-time vehicle detection and counting system, designed for traffic monitoring and smart city applications. The system tracks vehicles across video frames and counts them as they cross a configurable virtual detection line — supporting both inbound and outbound directions.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>🚗 Vehicle Detection: Detects cars, trucks, buses, and motorcycles</li>
        <li>📏 Line-Crossing Counter: Counts vehicles crossing a user-defined virtual line</li>
        <li>↕️ Directional Tracking: Distinguishes inbound vs. outbound traffic independently</li>
        <li>🎯 Multi-Class Support: Separate counters per vehicle class</li>
        <li>📊 Live Dashboard: Real-time count overlay on the video feed</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>inference pipeline via Ultralytics with optimized confidence thresholds</li>
        <li>Custom centroid-based line-crossing detection with directional logic</li>
        <li>Multi-object tracking with unique ID persistence across video frames</li>
        <li>Handles occlusion and overlapping vehicles with robust tracker configuration</li>
      </ul>
        `,
        role: `
      Solo CV Engineer:<br/>
      <ul>
        <li>🧠 Detection Pipeline: Configured for multi-class vehicle detection</li>
        <li>🔗 Tracker Integration: Implemented ID-based tracking across consecutive frames</li>
        <li>📐 Counting Logic: Built directional line-crossing detection algorithm</li>
        <li>🎨 Visualization: Rendered bounding boxes, IDs, and live counts with OpenCV</li>
      </ul>
        `,
        techStack: [
            'Python',
            'OpenCV',
            'Ultralytics',
            'Object Tracking',
            'Computer Vision',
        ],
        thumbnail: '/projects/thumbnail/81.png',
        longThumbnail: '/projects/long/82.png',
        images: [
        '/projects/images/81.png',
        '/projects/images/82.png',
    ],
    },
    // {
    //     title: 'Gesture-Controlled Snake Game',
    //     slug: 'snake-game-opencv',
    //     // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Snake-Game-Using-Open-CV',
    //     year: 2025,
    //     description: `
    //   A reimagination of the classic Snake game — entirely controlled by hand gestures via a webcam. No keyboard or mouse needed. The game tracks the player's index finger in real time and moves the snake accordingly, creating an immersive human-computer interaction experience.<br/><br/>

    //   Key Features:<br/>
    //   <ul>
    //     <li>🖐️ Gesture Control: Real-time hand tracking drives the snake — no keyboard required</li>
    //     <li>🐍 Classic Gameplay: Faithful snake mechanics with food collection and score tracking</li>
    //     <li>⚡ Real-Time Processing: Smooth 30+ FPS hand tracking with minimal latency</li>
    //     <li>📷 Webcam Input: Works with any standard webcam without special hardware</li>
    //     <li>🎮 Visual Feedback: Clean game rendering with live webcam overlay</li>
    //   </ul><br/>

    //   Technical Highlights:
    //   <ul>
    //     <li>Used MediaPipe Hands for accurate 21-keypoint hand landmark detection</li>
    //     <li>Extracted index finger tip coordinates to map to game control directions</li>
    //     <li>Built custom game loop in Python with OpenCV for rendering</li>
    //     <li>Implemented collision detection, food spawning, and score logic from scratch</li>
    //   </ul>
    //     `,
    //     role: `
    //   Solo Developer — designed and built the entire project:<br/>
    //   <ul>
    //     <li>🖐️ Hand Tracking: Integrated MediaPipe for real-time finger coordinate extraction</li>
    //     <li>🎮 Game Engine: Built the Snake game loop, collision system, and rendering with OpenCV</li>
    //     <li>🔗 Control Mapping: Mapped finger position to directional game input in real time</li>
    //     <li>🧪 Tuning: Calibrated detection sensitivity for smooth and responsive control</li>
    //   </ul>
    //     `,
    //     techStack: [
    //         'Python',
    //         'OpenCV',
    //         'MediaPipe',
    //         'Hand Tracking',
    //         'Computer Vision',
    //         'Game Development',
    //     ],
    //     thumbnail: '/projects/thumbnail/11.png',
    //     longThumbnail: '/projects/long/12.png',
    //     images: [
    //     '/projects/images/11.png',
    //     '/projects/images/12.png',
    // ],
    // },
    {
        title: 'Parking Spot Detector',
        slug: 'car-spot-detection-opencv',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Car-Spot-Detection-Using-OpenCV',
        year: 2025,
        description: `
      A smart parking management system that uses computer vision to detect available and occupied parking spaces in real time from a top-down camera feed. Each parking space is individually monitored, with live status updates rendered as a color-coded overlay on the video.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>🅿️ Space Detection: Automatically identifies individual parking spots from the camera view</li>
        <li>🟢🔴 Live Status: Green overlay for free spaces, red for occupied — updated every frame</li>
        <li>📊 Availability Counter: Displays total free and occupied spots in real time</li>
        <li>🎛️ Manual ROI Marking: Tool to define parking space regions on any parking lot layout</li>
        <li>📷 Video & Stream Support: Works with pre-recorded footage and live camera feeds</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Image preprocessing pipeline: grayscale → Gaussian blur → adaptive thresholding</li>
        <li>Pixel density analysis per ROI to determine occupancy state</li>
        <li>Pickle-based persistence for saving and loading parking space coordinates</li>
        <li>Configurable threshold sensitivity for different lighting and camera conditions</li>
      </ul>
        `,
        role: `
      Solo CV Engineer — complete system built independently:<br/>
      <ul>
        <li>🖊️ ROI Tool: Built a mouse-based interface for marking parking spaces on any layout</li>
        <li>⚙️ Detection Logic: Designed pixel-density thresholding algorithm for occupancy classification</li>
        <li>🎨 Visualization: Rendered color-coded overlays and live counters with OpenCV</li>
        <li>💾 Persistence: Implemented pickle serialization for layout save/load</li>
      </ul>
        `,
        techStack: [
            'Python',
            'OpenCV',
            'NumPy',
            'Image Processing',
            'Computer Vision',
        ],
        thumbnail: '/projects/thumbnail/101.png',
        longThumbnail: '/projects/long/102.png',
        images: [
        '/projects/images/101.png',
        '/projects/images/102.png',
    ],
    },
    {
        title: 'Tesla Stock Price Prediction (LSTM)',
        slug: 'stock-price-prediction-tesla',
        // liveUrl: 'https://github.com/Muhammad-Asim-Shaban/Stock-Price-Prediction-of-Tesla',
        year: 2024,
        description: `
      A deep learning model that predicts Tesla (TSLA) stock closing prices using Long Short-Term Memory (LSTM) neural networks. Trained on historical price data, the model captures long-range temporal dependencies in price sequences to generate future price forecasts.<br/><br/>

      Key Features:<br/>
      <ul>
        <li>📈 LSTM Forecasting: Sequential deep learning model trained on historical TSLA price data</li>
        <li>🔄 Sliding Window: Time-windowed input sequences for capturing price momentum patterns</li>
        <li>📊 Prediction Plots: Side-by-side visualization of actual vs. predicted closing prices</li>
        <li>📉 Loss Tracking: Training and validation loss curves for model evaluation</li>
        <li>🔢 Performance Metrics: RMSE and MAE reported for quantitative accuracy assessment</li>
      </ul><br/>

      Technical Highlights:
      <ul>
        <li>Built stacked LSTM architecture with Dropout layers to prevent overfitting</li>
        <li>Applied MinMaxScaler normalization for stable LSTM training</li>
        <li>Used 60-day sliding window to create input-output training pairs</li>
        <li>Visualized predictions vs. actuals with Matplotlib for clear performance evaluation</li>
      </ul>
        `,
        role: `
      Solo ML/DL Engineer:<br/>
      <ul>
        <li>📊 Data Preparation: Downloaded and preprocessed TSLA historical data with yfinance</li>
        <li>🧠 Model Design: Architected stacked LSTM with tuned layers, units, and Dropout</li>
        <li>🏋️ Training: Managed training loop with early stopping and learning rate scheduling</li>
        <li>📈 Evaluation: Compared predictions to ground truth with RMSE, MAE, and visual plots</li>
      </ul>
        `,
        techStack: [
            'Python',
            'LSTM',
            'TensorFlow / Keras',
            'NumPy',
            'Pandas',
            'Matplotlib',
            'yfinance',
        ],
        thumbnail: '',
        longThumbnail: '',
        images: [
        '',
        '',
    ],
    },
];
export const MY_EXPERIENCE = [
    {
        title: 'Software Engineer (Frontend)',
        company: 'Strativ AB',
        duration: 'Dec 2024 - Present',
    },
    {
        title: 'Frontend Developer',
        company: 'Epikcoders',
        duration: 'Oct 2023 - Nov 2024',
    },
    {
        title: 'Frontend Engineer',
        company: 'Anchorblock Technology',
        duration: 'Oct 2022 - Sep 2023',
    },
    {
        title: 'Frontend Developer (Part-time)',
        company: 'Branex IT',
        duration: 'Jan 2022 - Oct 2022',
    },
];
