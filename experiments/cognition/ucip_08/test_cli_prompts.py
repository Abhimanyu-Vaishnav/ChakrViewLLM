import subprocess

prompts = [
    "Hello ChakrView. How are you?",
    "What does a CPU do?",
    "नमस्ते! आप क्या कर सकते हैं?",
    "Bhai, machine learning ko simple language mein samjhao.",
    "Say exactly: The sky is blue.",
    "If I have 5 apples and give 2 to my friend, how many apples remain?"
]

cmd_input = "\n".join(prompts) + "\n/exit\n"
res = subprocess.run(
    [r".\.venv\Scripts\python.exe", "-m", "chakrview", "chat", "--checkpoint", r"releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt", "--memory", "off"],
    input=cmd_input,
    capture_output=True,
    text=True,
    cwd=r"D:\Project\ChakrView"
)

print(res.stdout)
