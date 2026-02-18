import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

print("✅ TensorFlow Version:", tf.__version__)
print("✅ Scenario Proto Loaded Successfully")

# Create a dummy scenario to test the Protobuf backend
s = scenario_pb2.Scenario()
s.scenario_id = "test_123"
print(f"✅ Protobuf Object Created: {s.scenario_id}")