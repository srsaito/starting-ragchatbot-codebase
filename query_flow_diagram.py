import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors
frontend_color = '#4CAF50'  # Green
api_color = '#2196F3'       # Blue
rag_color = '#FF9800'       # Orange
ai_color = '#9C27B0'        # Purple
search_color = '#F44336'    # Red
vector_color = '#795548'    # Brown
response_color = '#607D8B'  # Blue Grey

# Helper function to create fancy boxes
def create_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black',
                         linewidth=1.5, alpha=0.8)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            weight='bold', color='white', wrap=True)

# Helper function to create arrows
def create_arrow(ax, start, end, text="", offset=0.1):
    arrow = ConnectionPatch(start, end, "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc="black", ec="black", linewidth=2)
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + offset
        ax.text(mid_x, mid_y, text, ha='center', va='center',
                fontsize=9, weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', edgecolor='gray', alpha=0.9))

# Title
ax.text(5, 13.5, 'RAG Chatbot: User Query Flow (Frontend → Backend)', 
        ha='center', va='center', fontsize=18, weight='bold')

# 1. Frontend Layer
create_box(ax, 0.5, 11.5, 2, 1.5, 'FRONTEND\n(script.js)\n\nUser Input\n→ sendMessage()', frontend_color, 10)

# 2. API Gateway
create_box(ax, 3.5, 11.5, 2, 1.5, 'API GATEWAY\n(app.py)\n\nPOST /api/query\n→ FastAPI', api_color, 10)

# 3. RAG System
create_box(ax, 6.5, 11.5, 3, 1.5, 'RAG SYSTEM\n(rag_system.py)\n\nOrchestrates query\nManages session', rag_color, 10)

# 4. AI Generator
create_box(ax, 1, 9, 3, 1.5, 'AI GENERATOR\n(ai_generator.py)\n\nClaude AI + Tools\nDecides: Search or Knowledge?', ai_color, 10)

# 5. Search Tools (when needed)
create_box(ax, 5.5, 9, 3.5, 1.5, 'SEARCH TOOLS\n(search_tools.py)\n\nCourseSearchTool\nSemantic search execution', search_color, 10)

# 6. Vector Store
create_box(ax, 5.5, 6.5, 3.5, 1.5, 'VECTOR STORE\n(vector_store.py)\n\nChromaDB\nSimilarity search', vector_color, 10)

# 7. Response Assembly
create_box(ax, 1, 4, 8, 1.5, 'RESPONSE ASSEMBLY\nAI synthesizes answer → Sources collected → Session updated → JSON response', response_color, 10)

# 8. Frontend Display
create_box(ax, 3, 1.5, 4, 1.5, 'FRONTEND DISPLAY\n(script.js)\n\nMarkdown rendering + Sources\nUpdate conversation', frontend_color, 10)

# Add arrows for the flow
create_arrow(ax, (2.5, 12.2), (3.5, 12.2), "POST request")
create_arrow(ax, (5.5, 12.2), (6.5, 12.2), "query()")
create_arrow(ax, (7.5, 11.5), (2.5, 10.5), "generate_response()")

# Conditional path - AI to Search
create_arrow(ax, (2.5, 9), (5.5, 9.7), "IF course\nquery")

# Search to Vector Store
create_arrow(ax, (7.2, 9), (7.2, 8), "semantic\nsearch")

# Vector Store back to Search
create_arrow(ax, (6.8, 6.5), (6.8, 9), "results")

# All paths converge to Response Assembly
create_arrow(ax, (2.5, 9), (3, 5.5), "answer")
create_arrow(ax, (7, 9), (7, 5.5), "sources")

# Response back to frontend
create_arrow(ax, (5, 4), (5, 3), "JSON\nresponse")

# Add side annotations
ax.text(0.2, 10, '1', ha='center', va='center', fontsize=16, weight='bold', 
        bbox=dict(boxstyle="circle", facecolor='yellow'))
ax.text(0.2, 8, '2', ha='center', va='center', fontsize=16, weight='bold',
        bbox=dict(boxstyle="circle", facecolor='yellow'))
ax.text(0.2, 6, '3', ha='center', va='center', fontsize=16, weight='bold',
        bbox=dict(boxstyle="circle", facecolor='yellow'))
ax.text(0.2, 4, '4', ha='center', va='center', fontsize=16, weight='bold',
        bbox=dict(boxstyle="circle", facecolor='yellow'))
ax.text(0.2, 2, '5', ha='center', va='center', fontsize=16, weight='bold',
        bbox=dict(boxstyle="circle", facecolor='yellow'))

# Add legend for decision point
ax.text(9.5, 8.5, 'DECISION\nPOINT:\n\nGeneral question?\n→ Use AI knowledge\n\nCourse question?\n→ Search database', 
        ha='center', va='center', fontsize=9, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='orange'))

# Add key files referenced
ax.text(0.5, 0.5, 'Key Files:\n• frontend/script.js:45 (sendMessage)\n• backend/app.py:56 (/api/query)\n• backend/rag_system.py:102 (query)\n• backend/ai_generator.py:43 (generate_response)\n• backend/search_tools.py:20 (CourseSearchTool)', 
        ha='left', va='bottom', fontsize=8, style='italic',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

plt.tight_layout()
plt.savefig('/Users/stevensaito/ML/starting-ragchatbot-codebase/query_flow_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Query flow diagram saved as 'query_flow_diagram.png'")