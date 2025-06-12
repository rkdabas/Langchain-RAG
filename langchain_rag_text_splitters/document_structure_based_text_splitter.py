from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        self.result = a + b
        return self.result
    
    def subtract(self, a, b):
        self.result = a - b
        return self.result
    
    def multiply(self, a, b):
        self.result = a * b
        return self.result
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        self.result = a / b
        return self.result

def main():
    calc = Calculator()
    print("Addition:", calc.add(10, 5))
    print("Subtraction:", calc.subtract(10, 5))
    print("Multiplication:", calc.multiply(10, 5))
    print("Division:", calc.divide(10, 5))

if __name__ == "__main__":
    main()


"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=20,
) 

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])
print(chunks[1])