#!groovy

parallel (

"gin" : {node ('gin') 
  {
    stage ('gin_build')
    {
      deleteDir()
      checkout scm
      sh "./build.sh $WORKSPACE $NODE_NAME"
    }

    stage ('gin_run')
    {
      sh "./src/mem"
      sh "./success.sh 2 gin openfpm_devices"
    }
  }},

"nyu" : {node ('nyu')
  {
    stage ('nyu_build')
    {
      deleteDir()
      checkout scm
      sh "./build.sh $WORKSPACE $NODE_NAME"
    }

    stage ('nyu_run')
    {
      sh "./src/mem"
      sh "./success.sh 2 nyu openfpm_devices"
    }
  }},

"sb15" : {node ('sbalzarini-mac-15')
  {
    env.PATH = "/usr/local/bin:${env.PATH}"
    stage ('sb15_build')
    {
      deleteDir()
      checkout scm
      sh "./build.sh $WORKSPACE $NODE_NAME"
    }

    stage ('sb15_run')
    {
      sh "./src/mem"
      sh "./success.sh 2 sbalzarini-mac-15 openfpm_devices"
    }
  }}


)

