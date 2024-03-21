import toml
import os
import sys
import argparse
from abc import ABC, abstractmethod

class TomlSanityCheck:
    def __init__(self, file_path, verbose = False):
        self.config_file = toml.load(file_path)
# Make sure that dictionary has correct formatting. For reference, the file formatted as follows:

# Each parameter is formatted like: [XXX.YYY] where XXX refers to the group of the parameter and YYY refers to the name
# Each parameter has exactly two key:value pairs.
#   The first is the actual value of the parameter (a path, a number, a string etc.)
#   The second is the sanity check/constraint on the parameter. See TomlSanityCheck.py for a current list of supported constraints

        for section in self.config_file:
            if verbose:
                print(section)
            current_params = self.config_file[section]
            for parameter in current_params:
                if verbose:
                    print("\t"+str(parameter))
                keys = current_params[parameter]
# We only allow 2 keys per parameter (ostenstibly 'value' and 'constraint')
                if len(keys) != 2:
                    err_msg = "Invalid number of keys for parameter '"+str(parameter)
                    err_msg += "' under section '"+str(section)+"'"
                    raise Exception(err_msg)
# check that keys are either 'value' or 'constraint'
                for key in keys:
                    if verbose:
                        print("\t\t"+str(keys))
                    if ((key.lower() != "value") and (key.lower() != "constraint")):
                        err_msg = "Key named '"+ str(key) +"' in parameter '" +str(parameter)
                        err_msg += "' under section '"+str(section) + "' is neither 'value' nor 'constraint'"
                        raise Exception(err_msg)

    def validate(self, sections=[]):
        if (len(sections)==0):
            sections = [sec for sec in self.config_file]
        for section_name in sections:
            try:
                section = self.config_file[section_name]
            except:
                raise Exception(section +" is not found in config file!")
            for parameter in section:
                val = section[parameter]["value"]
                constraint_name = section[parameter]["constraint"]
                constraint = None
                match constraint_name:
                    case "None":
                        constraint = NoConstraint(val)
                    case "NonEmptyString":
                        constraint = NonEmptyString(val)
                    case "SPosInt":
                        constraint = StrictPositiveInt(val)
                    case "PosInt":
                        constraint = PositiveInt(val)
                    case "SPosFloat":
                        constraint = StrictPositiveFloat(val)
                    case "PosFloat":
                        constraint = PositiveFloat(val)
                    case "ValidFolder":
                        constraint = ValidFolder(val)
                    case "DetectorGeometry":
                        constraint = DetectorGeometry(val)
                    case "Boolean":
                        constraint = BooleanCheck(val)
                    case "TrainingTarget":
                        constraint = TrainingTarget(val)
                    case "TrainingModel":
                        constraint = TrainingModel(val)
                if not constraint:
                    err_msg = "Constraint '"+constraint_name + "' is currently not defined."
                    err_msg += " Did you create a inherited class, and add the constraint to the above match statement?"
                    raise Exception(err_msg)
                if not constraint.validate():
                    err_msg = "parameter '"+str(parameter)+"' with value '"+str(val)+"'"
                    err_msg += " doesn't satisfy constraint '"+str(constraint_name)+"'"
                    raise Exception(err_msg)

    def return_config(self):
        return self.config_file
    
    def gen_bash_variables(self,sections = []):
        output = ""
        if sections== []:
            groups = [g for g in self.config_file]
        else:
            groups = sections
        for group in groups:
            try:
                section = self.config_file[group]
            except:
                raise Exception(section +" is not found in config file!")
            for argument in section:
                bash_var = group+"_"+str(argument)+"="+str(self.config_file[group][argument]["value"])+"\n"
                output += bash_var
        return output
# Abstract base class from which all other constraints are derived from
class TOMLParameterConstraint(ABC):
    def __init__(self,parameter_value):
        self.value = parameter_value
# We expect the method to return a boolean stating weather self.value satisfies the constraint
    @abstractmethod
    def validate(self):
        pass

class NoConstraint(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(NonEmptyString, self).__init__(parameter_value)
    def validate(self):
        return True

class NonEmptyString(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(NonEmptyString, self).__init__(parameter_value)
    def validate(self):
        if not isinstance(self.value, str):
            return False
        if len(self.value) == 0:
            return False
        return True

class StrictPositiveInt(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(StrictPositiveInt, self).__init__(parameter_value)
    def validate(self):
        if not isinstance(self.value,int):
            return False
        try:
            ge = (self.value > 0)
        except:
            return False
        return ge

class PositiveInt(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(PositiveInt, self).__init__(parameter_value)
    def validate(self):
        if not isinstance(self.value,int):
            return False
        try:
            ge = (self.value >= 0)
        except:
            return False
        return ge

class StrictPositiveFloat(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(StrictPositiveFloat, self).__init__(parameter_value)
    def validate(self):
        if not isinstance(self.value,float):
            return False
        try:
            ge = (self.value > 0)
        except:
            return False
        return ge

class PositiveFloat(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(PositiveFloat, self).__init__(parameter_value)
    def validate(self):
        if not isinstance(self.value,float):
            return False
        try:
            ge = (self.value >= 0)
        except:
            return False
        return ge

class ValidFolder(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(ValidFolder, self).__init__(parameter_value)
    def validate(self):
        if os.path.isdir(self.value):
            return True
        return False

class DetectorGeometry(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(DetectorGeometry, self).__init__(parameter_value)
    def validate(self):
        valid_geometries = ["cube","flat"]
        lowered = self.value.lower()
        if lowered not in valid_geometries:
            return False
        return True

class BooleanCheck(TOMLParameterConstraint):
# Checks for non-empty string as value
    def __init__(self, parameter_value):
        super(BooleanCheck, self).__init__(parameter_value)
    def validate(self):
        if isinstance(self.value,bool):
            return True
        return False

class TrainingTarget(TOMLParameterConstraint):
    def __init__(self, parameter_value):
        super(TrainingTarget,self).__init__(parameter_value.lower().strip())
    def validate(self):
        if((self.value == "class") or (self.value == "energy")):
            return True
        else:
            return False

class TrainingModel(TOMLParameterConstraint):
    def __init__(self, parameter_value):
        super(TrainingModel,self).__init__(parameter_value.lower().strip())
    def validate(self):
        if((self.value == "simple") or (self.value == "cnn")):
            return True
        else:
            return False

if __name__ =="__main__":
# By Default, converts toml file to a list of bash variables
    parser = argparse.ArgumentParser(prog='CreateMCNNData')
    parser.add_argument('GenDataTOML',help="Path to .toml config file to generate data")
    parser.add_argument("-s", "--section" ,type=str, help="Comma delimited list stating which section(s) to read in. If nothing, just reads and validates everything")
    args = parser.parse_args()
    if args.section:
    # Splits section argument into constituent parts, removes all whitespace in each part, then creates list
        sections = [''.join(item.split()) for item in args.section.split(',')]
    else:
        sections = []
    try:
        s = TomlSanityCheck(args.GenDataTOML)
        s.validate(sections)
    except Exception as err:
        print("Parsing Failed!")
        print(err)
        sys.exit(1)
    # Assuming all went well, print out a string containing bash variables
    print(s.gen_bash_variables(sections=sections))
