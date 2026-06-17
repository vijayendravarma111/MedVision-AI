import gradio as gr

html = """
<div style="background:#020B2D;
padding:40px;
min-height:100vh;
color:white;
font-family:Segoe UI;">

<h1 style="text-align:center;
font-size:60px;">
MedVision AI
</h1>

<h3 style="text-align:center;
color:#cfcfcf;">
Smarter Diagnosis Through Imaging
</h3>

<hr>

<div style="
display:flex;
justify-content:center;
gap:30px;
margin-top:60px;
flex-wrap:wrap;
">

<div style="
background:#10182D;
padding:30px;
width:320px;
border-radius:20px;
text-align:center;
">

<h2>BrainScan Insight</h2>

<p>
Detects brain tumors from MRI scans.
</p>

<a href="http://127.0.0.1:7860"
target="_blank">

<button style="
padding:12px 25px;
border:none;
border-radius:10px;
background:#4F46E5;
color:white;
font-size:18px;
cursor:pointer;
">

Launch Tool

</button>

</a>

</div>

<div style="
background:#10182D;
padding:30px;
width:320px;
border-radius:20px;
text-align:center;
">

<h2>BoneExpert AI</h2>

<p>
Detects bone fractures using X-rays.
</p>

<a href="http://127.0.0.1:7862"
target="_blank">

<button style="
padding:12px 25px;
border:none;
border-radius:10px;
background:#4F46E5;
color:white;
font-size:18px;
cursor:pointer;
">

Launch Tool

</button>

</a>

</div>

<div style="
background:#10182D;
padding:30px;
width:320px;
border-radius:20px;
text-align:center;
">

<h2>ChestGuard AI</h2>

<p>
Detects chest diseases from X-rays.
</p>

<a href="http://localhost:8501"
target="_blank">

<button style="
padding:12px 25px;
border:none;
border-radius:10px;
background:#4F46E5;
color:white;
font-size:18px;
cursor:pointer;
">

Launch Tool

</button>

</a>

</div>

</div>

</div>
"""

with gr.Blocks() as app:

    gr.HTML(html)

app.launch(
    server_port=7865
)