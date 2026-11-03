"""Agent 03: Reflexion. Try, test, write a verbal lesson, retry with the
lessons in memory. Lessons 14.03, 14.05."""
from .llm import ask


def run(task, test, tries=3):
    """test(answer) -> (passed: bool, feedback: str)"""
    lessons = []
    for i in range(tries):
        answer = ask(f"Task: {task}\nLessons from earlier failures:\n"
                     + "\n".join(lessons or ["none"]))
        ok, feedback = test(answer)
        print(f"try {i + 1}: {'pass' if ok else 'fail'} {feedback[:100]}")
        if ok:
            return answer, lessons
        lessons.append(ask(f"Answer:\n{answer}\nFeedback: {feedback}\n"
                           "In one sentence, what should the next attempt do differently?"))
    return answer, lessons
