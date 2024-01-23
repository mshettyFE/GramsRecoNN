import toml
import os
from abc import ABC, abstractmethod

class TomlSanityCheck:
    def __init__(self, file_path, verbose = False):
        try:
            self.config_file = toml.load(file_path)
        except:
            raise Exception("Couldn't parse "+str(file_path))
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

    def validate(self):
        for section_name in self.config_file:
            section = self.config_file[section_name]
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

if __name__ =="__main__":
    base = os.getcwd()
    s = TomlSanityCheck(os.path.join(base,"MCTruth","MCTruth.toml"))
    s.validate()