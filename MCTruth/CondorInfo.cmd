executable     = /nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/MCTruth/CondorSetup.sh
transfer_input_files = /nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/MCTruth/CreateMCNNData.py,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/TomlSanityCheck.py,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/GramsSimWork.tar.gz,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/MCTruth/Config.toml,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/gdml,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/MCTruth/DataGen.sh


arguments = $(Process)

universe = vanilla
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

requirements = ( Arch == "X86_64" )

output         = temp-$(Process).out
error          = temp-$(Process).err
log            = temp-$(Process).log
initialdir=/nevis/milne/files/ms6556/BleekerData/GramsMLRecoData/CondorTest
