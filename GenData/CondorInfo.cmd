executable     = /nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/GenData/CondorSetup.sh
transfer_input_files = /nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/GenData/CreateData.py,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/TomlSanityCheck.py,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/GramsSimWork.tar.gz,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/GenData/Config.toml,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/gdml,/nevis/milne/files/ms6556/BleekerShareBuild/NNGramsReco/GenData/DataGen.sh


arguments = $(Process)

universe = vanilla
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

# machine boolean temporary hack since amsterdan running CentOs. Workarout for that is just not submitting to amsterdam
requirements =( ( Arch == "X86_64" ) && ( machine != "amsterdam.nevis.columbia.edu" ))

output         = temp-$(Process).out
error          = temp-$(Process).err
log            = temp-$(Process).log
initialdir=/nevis/milne/files/ms6556/RiversideData/GramsMLRecoData/Train
