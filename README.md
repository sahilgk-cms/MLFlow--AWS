# MLFlow--AWS
Use MLFlow with Amazon EC2 + S3

## Create a S3 bucket
- Create a bucket in Amazon **S3** to store all artifact data.
- Artifact name: **mlflow-artifacts-sahil**

## Create an EC2 instance
- Create am Amazon **EC2** instance.
- Select the instance type to be **t3.medium** (4GB RAM, 2 vCPUs). **MLFlow** requires min **4GB RAM**. Anything less than that leads to performance issues. 
- Create a new **IAM role** with policy **AmazonS3FullAccess** and assign it to the **EC2** instance.
- In **Security**, allow port **5000** in **Inbound rules**. This is for running **MLFlow**.
- Once the instance is created, download **.pem** for connecting it with SSH.

```
ssh -i "mlflow-sk.pem" ubuntu@ec2-35-154-73-149.ap-south-1.compute.amazonaws.com
```

## Installing AWS CLI:
Run these commands to install aws cli inside ec2 instance
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip -y
unzip awscliv2.zip
sudo ./aws/install
```


