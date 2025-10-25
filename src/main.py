import os
from enum import Enum
from typing import List, Dict, Optional, Union
import pickle
import numpy as np
from pydantic import BaseModel

try:
    import tensorflow as tf
except ImportError:
    tf = None

class ModelType(Enum):
    TENSORFLOW = "tensorflow"
    PICKLE = "pickle"

class LocalModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.model_type = self._determine_model_type()
        self.load_model()

    def _determine_model_type(self) -> ModelType:
        if self.model_path.endswith('.h5'):
            if tf is None:
                raise ImportError("TensorFlow is required for .h5 models. Install it with: pip install tensorflow")
            return ModelType.TENSORFLOW
        elif self.model_path.endswith('.pkl'):
            return ModelType.PICKLE
        else:
            raise ValueError("Unsupported model format. Use either .h5 or .pkl files")

    def load_model(self):
        if self.model_type == ModelType.TENSORFLOW:
            self.model = tf.keras.models.load_model(self.model_path)
        else:  # PICKLE
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

    def generate_response(self, input_text: str) -> str:
        if self.model is None:
            raise ValueError("Model not loaded")

        # Convert input text to model-appropriate format
        # This is a placeholder - modify according to your model's requirements
        if self.model_type == ModelType.TENSORFLOW:
            # Example preprocessing for tensorflow model
            # Modify this according to your model's input requirements
            return self._generate_tensorflow_response(input_text)
        else:
            # Example preprocessing for pickle model
            # Modify this according to your model's input requirements
            return self._generate_pickle_response(input_text)

    def _generate_tensorflow_response(self, input_text: str) -> str:
        # Implement your tensorflow model inference here
        # This is just an example - modify according to your model
        try:
            # Example: Convert text to sequence, pad, predict, etc.
            # prediction = self.model.predict(processed_input)
            # return self._process_prediction(prediction)
            return f"TensorFlow model response for: {input_text}"
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _generate_pickle_response(self, input_text: str) -> str:
        # Implement your pickle model inference here
        # This is just an example - modify according to your model
        try:
            # Example: Process input according to your model's requirements
            # prediction = self.model.predict([input_text])
            # return self._process_prediction(prediction)
            return f"Pickle model response for: {input_text}"
        except Exception as e:
            return f"Error generating response: {str(e)}"

class ThoughtType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"

class Thought(BaseModel):
    type: ThoughtType
    content: str

class Agent:
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or """You are a helpful autonomous agent.
Your task is to help users accomplish their goals through careful thinking and action.
Always follow this process:
1. Think about what you know and what you need to find out
2. Decide on the next action
3. Execute the action and observe the results
4. Continue this process until you reach a final answer
Format your responses as one of these types:
THOUGHT: for your reasoning process
ACTION: for actions you want to take
OBSERVATION: for results of actions
FINAL_ANSWER: when you have completed the task"""
        self.conversation_history = []

    def _create_messages(self) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages

    def _parse_response(self, response: str) -> Thought:
        for thought_type in ThoughtType:
            prefix = f"{thought_type.value.upper()}: "
            if response.upper().startswith(prefix):
                content = response[len(prefix):].strip()
                return Thought(type=thought_type, content=content)

        # Default to treating it as a thought if no prefix is found
        return Thought(type=ThoughtType.THOUGHT, content=response)

    def __init__(self, model_path: str = None, system_prompt: str = None):
        self.system_prompt = system_prompt or """You are a helpful autonomous agent.
Your task is to help users accomplish their goals through careful thinking and action.
Always follow this process:
1. Think about what you know and what you need to find out
2. Decide on the next action
3. Execute the action and observe the results
4. Continue this process until you reach a final answer
Format your responses as one of these types:
THOUGHT: for your reasoning process
ACTION: for actions you want to take
OBSERVATION: for results of actions
FINAL_ANSWER: when you have completed the task"""
        self.conversation_history = []
        self.model = LocalModel(model_path) if model_path else None

    def think(self, user_input: str) -> List[Thought]:
        thoughts = []
        max_steps = 10  # Prevent infinite loops

        if not self.model:
            return [Thought(
                type=ThoughtType.FINAL_ANSWER,
                content="No model loaded. Please provide a model path (.h5 or .pkl file)"
            )]

        try:
            # Add system prompt and user input to context
            context = f"{self.system_prompt}\n\nUser: {user_input}"

            # Generate initial response
            response_text = self.model.generate_response(context)
            thought = self._parse_response(response_text)
            thoughts.append(thought)

            # Continue thinking if needed
            for _ in range(max_steps - 1):
                if thought.type == ThoughtType.FINAL_ANSWER:
                    break

                # Update context with previous thought
                context = f"Previous thought: {thought.content}\nContinue thinking..."
                response_text = self.model.generate_response(context)
                thought = self._parse_response(response_text)
                thoughts.append(thought)

        except Exception as e:
            print(f"Error during thinking process: {str(e)}")
            thoughts.append(Thought(
                type=ThoughtType.FINAL_ANSWER,
                content=f"I encountered an error: {str(e)}"
            ))

        return thoughts

def main():
    try:
        # Example paths - replace with your actual model paths
        tf_model_path = "path/to/your/model.h5"  # For TensorFlow model
        pkl_model_path = "path/to/your/model.pkl"  # For pickle model

        # Create agent with your preferred model
        print("Choose your model type:")
        print("1. TensorFlow model (.h5)")
        print("2. Pickle model (.pkl)")
        choice = input("Enter 1 or 2: ")

        model_path = tf_model_path if choice == "1" else pkl_model_path
        agent = Agent(model_path=model_path)

        # Run a test query
        print("\nRunning test query...")
        query = input("Enter your question: ")
        thoughts = agent.think(query)

        print("\nThoughts:")
        for thought in thoughts:
            print(f"{thought.type.value.upper()}: {thought.content}")

    except Exception as e:
        print(f"\nError in main: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your model file exists and is accessible")
        print("2. For .h5 files, ensure TensorFlow is installed")
        print("3. Check if your model format matches the file extension")
        print("4. Verify that your model is compatible with the input format")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

