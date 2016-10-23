#!groovy
node ('gin') 
{
  stage 'build'
  checkout scm
  sh "./build_device.sh $WORKSPACE $NODE_NAME"

  stage 'run'
  sh "./src/mem"
}

