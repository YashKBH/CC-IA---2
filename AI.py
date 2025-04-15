import boto3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Initialize AWS CloudWatch and EC2 clients
cloudwatch = boto3.client('cloudwatch')
ec2 = boto3.client('ec2')

# Config: instance ID and threshold
INSTANCE_ID = 'i-0123456789abcdef0'
CPU_THRESHOLD = 70  # percent

# Step 1: Get CPU Utilization Metrics
def get_cpu_utilization(instance_id, period=300, duration_minutes=60):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=duration_minutes)

    metrics = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=period,
        Statistics=['Average']
    )

    data = metrics['Datapoints']
    if not data:
        return None

    df = pd.DataFrame(data)
    df.sort_values('Timestamp', inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df['Average']

# Step 2: Predict CPU Usage using Linear Regression
def predict_future_usage(cpu_data):
    cpu_data = cpu_data.dropna()
    X = np.arange(len(cpu_data)).reshape(-1, 1)
    y = cpu_data.values

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.array([[len(cpu_data) + 1]])
    predicted_cpu = model.predict(future_index)[0]
    return predicted_cpu

# Step 3: Scale up/down based on prediction
def manage_resources(predicted_cpu):
    if predicted_cpu > CPU_THRESHOLD:
        print(f"Predicted CPU {predicted_cpu:.2f}% > threshold. Starting another instance...")
        # Add automation logic: e.g., start new instance
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
    else:
        print(f"CPU usage is fine: {predicted_cpu:.2f}%. No action needed.")

# Run the automation
cpu_data = get_cpu_utilization(INSTANCE_ID)
if cpu_data is not None:
    predicted = predict_future_usage(cpu_data)
    manage_resources(predicted)
else:
    print("No CPU data available.")
