import os
import re
import PyPDF2


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.prompts import PromptTemplate

class PDFAcademicAdvisor:
    def __init__(self,
                 model_name: str = "llama3.2:1b",
                 collection_name: str = "academic_docs"):
        """
        Initializing the RAG system with PDF documents
        """
        # Set PDF directory relative to the script location
        self.pdf_directory = os.path.join(os.path.dirname(__file__), "data")

        # Embedding and LLM setup
        try:
            self.embeddings = OllamaEmbeddings(model=model_name)
            self.llm = Ollama(model=model_name)
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            print("Ensure Ollama is running and the model is pulled")
            exit(1)

        # Vector store setup
        self.persist_directory = "./chroma_db"
        self.collection_name = collection_name

        # Defining the prompt template
        self.qa_template = """You are an academic advisor for the Master of Science in Computer Science (MSCS) program at Cal State University Fullerton. 
Your primary goal is to provide accurate, helpful, and student-focused guidance for MSCS students. 
Follow these key principles:

1. Provide precise and official information about the MSCS program
2. Offer clear, constructive advice about course selection, degree requirements, and academic planning
3. Explain program policies, graduation requirements, and academic procedures
4. Be supportive and professional in all interactions
5. If a query is beyond the scope of the available information, advise the student to contact the official academic advising office

Always base your responses on the official documents and program guidelines. 
Respond as a knowledgeable, empathetic academic advisor who wants to support student success.

Use the following pieces of context to answer the question. 
If you don't know the answer, say so politely and suggest contacting the official academic advising office.

Context:
{context}

Question: {question}

Helpful Answer:"""

        self.prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )

        # Ensure persistent directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

    def extract_text_from_pdfs(self) -> List[str]:
        """
        Extract text from all PDFs in the specified directory
        """
        texts = []

        # Print directory contents for debugging
        print("PDFs in directory:", os.listdir(self.pdf_directory))

        # Iterate through all files in the directory
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)

                try:
                    with open(pdf_path, 'rb') as file:
                        # Create PDF reader object
                        pdf_reader = PyPDF2.PdfReader(file)

                        # Extract text from each page
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() + "\n"

                        # Clean up text (remove extra whitespaces, newlines)
                        pdf_text = re.sub(r'\s+', ' ', pdf_text).strip()
                        texts.append(pdf_text)
                        print(f"Processed {filename}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return texts

    def process_documents(self, texts: List[str]) -> List[str]:
        """
        Split documents into chunks for embedding
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_text("\n\n".join(texts))

    def create_vector_store(self, texts: List[str]):
        """
        Create a vector store from processed documents
        """
        return Chroma.from_texts(
            texts,
            self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

    def setup_retrieval_qa(self):
        """
        Set up the retrieval QA chain
        """
        vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        retriever = vectordb.as_retriever(
            search_kwargs={"k": 3}  # retrieves top 3 most relevant documents
        )



        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )

    def run_advisor(self):
        """
        Main method to run the academic advisor
        """
        # Extract and process PDF documents
        extracted_texts = self.extract_text_from_pdfs()
        print(f"Extracted text from {len(extracted_texts)} PDFs")

        if not extracted_texts:
            print("No texts extracted. Check PDF files and directory.")
            return

        processed_texts = self.process_documents(extracted_texts)
        print(f"Processed into {len(processed_texts)} text chunks")

        self.create_vector_store(processed_texts)

        # Setup QA chain
        qa_chain = self.setup_retrieval_qa()

        # Interactive loop
        print("🎓 Academic Advisor RAG System 🎓")
        print("Type 'exit' to quit.")

        while True:
            query = input("\nAsk your academic advising question: ")

            if query.lower() == 'exit':
                break

            try:
                result = qa_chain(query)
                print("\n📚 Answer:", result['result'])

                # Show source documents
                print("\n🔍 Sources:")
                for doc in result.get('source_documents', []):
                    print(f"- {doc.page_content[:250]}...")

            except Exception as e:
                print(f"Error processing query: {e}")


def main():
    advisor = PDFAcademicAdvisor()
    advisor.run_advisor()


if __name__ == "__main__":
    main()