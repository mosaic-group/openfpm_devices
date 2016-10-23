#!groovy

parallel (

"gin" : {node ('gin') 
  {
    stage ('gin_build')
    {
      checkout scm
      sh "./build.sh $WORKSPACE $NODE_NAME"
    }

    stage ('gin_run')
    {
      sh "./src/mem"
    }
  }},

"nyu" : {node ('nyu')
  {
    stage ('nyu_build')
    {
      checkout scm
      sh "./build.sh $WORKSPACE $NODE_NAME"
    }

    stage ('nyu_run')
    {
      sh "./src/mem"
    }
  }},

"sb15" : {node ('sbalzarini-mac-15')
  {
    env.PATH = "/usr/local/bin:${env.PATH}"
    stage ('sb15_build')
    {
      checkout scm
      sh "./build.sh $WORKSPACE $NODE_NAME"
    }

    stage ('sb15_run')
    {
      sh "./src/mem"
    }
  }}


)

