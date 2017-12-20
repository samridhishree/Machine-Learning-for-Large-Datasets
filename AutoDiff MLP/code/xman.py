# Expression manager.  You can subclass this to defined a network
# architecture, eg
#
# class f(XManFunctions): 
#    ....
#
# class LogisticRegression(XMan):
#    input = Input()
#    weights = Param()
#    output = f.logistic( input * weights )
#
# This will internally build an expression graph, made up of Registers
# and Operations.  Registers are like variables; operations are like
# function calls.
# 
# Each register has a 'role': it is either an input, a parameter, or
# an operationOutput.  If reg is an operationOutput, then the field
# reg.definedAs points to the Operation "op" which defines reg, and
# the field op.outputReg points back to reg.
#
# The static functions under XManFunctions and its subclasses (like
# f.logistic) should return an operationOutput register, which is
# linked as described above: i.e., it is definedAs some appropriate
# Operation's outputReg. Some python operators on registers are
# overloaded as "f."  functions: eg r1*r2 means f.dot(r1,r2) and r1+r2
# means f.add(r1,r2).
#
# Each Operation contains an OperationFunction, a list of arguments
# (which are registers), and a output.
#
# Every register has a string 'name'. If some python variable is bound
# to a register (like the class variables 'input', 'weights', and
# 'output' in the example) then when xman.setup() is called,
# reflection is used to assign that string to a register: so 
# after
# 
#  lr = LogisticRegression()
#  lr.setup()
#
# then lr.input.name will be 'input', and so on.  The setup() routine
# also traverses the expression tree and assigns unique names
# (z1,z2,...) to all registers reachable from class or instance
# variables.
#


class Operation(object):
    """ An operation encodes a single step of a differentiable
    computation, eg y=f(x1,...,xk). It contains a function, arguments,
    and a pointer to the register that is defined as the output of
    this operation.
    """

    def __init__(self,fun,*args):
        self.fun = fun
        self.args = args
        self.outputReg = None

    def asStringTuple(self):
        """ Return a nested tuple of encoding the operation y=f(x1,...xk) of
        the form (y,f,(x1,...,xk)) where y,f, and x1...xk are strings.
        """
        dstName = self.outputReg.name if (self.outputReg and self.outputReg.name) else "???"
        argNames = map(lambda reg:reg.name or "???", self.args)
        return (dstName,self.fun,argNames)

    def __str__(self):
        """ Human readable representation """
        (dstName,fun,argNames) = self.asStringTuple()
        return dstName + " = f." + fun + "(" + ",".join(argNames) + ")"


class Register(object):
    """ Registers are like variables - they are used as the inputs to and
    outputs of Operations.  The 'name' of each register should be unique,
    as it will be used as a key in storing outputs, and 
    """
    _validRoles = set("input param operationOutput".split())

    def __init__(self,name=None,role=None,default=None):
        assert role in Register._validRoles
        self.role = role
        self.name = name
        self.definedAs = None
        self.default = default

    def inputsTo(self):
        """ Trace back through the definition of this register, if it exists,
        to find a list of all other registers that this register
        depends on.  This method is needed to find the
        operationSequence that is needed to construct the value of a
        register, and also to assign names to otherwise unnamed
        registers.
        """
        if self.definedAs:
            assert isinstance(self.definedAs,Operation)
            return self.definedAs.args
        else:
            return []
    # operator overloading 
    def __add__(self,other):
        return XManFunctions.add(self,other)
    def __sub__(self,other):
        return XManFunctions.subtract(self,other)
    def __mul__(self,other):
        return XManFunctions.mul(self,other)

class XManFunctions(object):
    """ Encapsulates the static methods that are used in a subclass of
    XMan.  Each of these generates an OperationOutput register that is
    definedBy an Operation, with the operations outputReg field
    pointing back to the register.

    You will usually subclass this so you can
    add your own functions, and give the subclass
    a short name

    """

    @staticmethod
    def input(name=None, default=None):
        return Register(name=name, role='input',default=default)
    @staticmethod
    def param(name=None, default=None):
        return Register(name=name, role='param',default=default)

    @staticmethod
    def add(a,b):
        return XManFunctions.registerDefinedByOperator('add',a,b)
    @staticmethod
    def subtract(a,b):
        return XManFunctions.registerDefinedByOperator('subtract',a,b)
    @staticmethod
    def mul(a,b):
        return XManFunctions.registerDefinedByOperator('mul',a,b)

    @staticmethod
    def registerDefinedByOperator(fun,*args):
        """ Create a 
        """
        assert all(map(lambda a:isinstance(a,Register), args)), 'arguments should all be registers: %r %r' % (fun,args)
        reg = Register(role='operationOutput')
        op = Operation(fun,*args)
        reg.definedAs = op
        op.outputReg = reg
        return reg

class XMan(object):

    def __init__(self):
        self._nextTmp = 1  
        self._setupComplete = False
        self._registers = {}

    def setup(self):
        """ Must call this before you do any other operations with an
        expression manager """
        # use available python variable names for register names
        for regName,reg in self.namedRegisterItems():
            if not reg.name:
                reg.name = regName
                self._registers[regName] = reg
        # give names to any other registers used in subexpressions
        def _recursivelyLabelUnnamedRegisters(reg, seen=None):
            if seen is None: seen = set()
            if not reg.name:
                reg.name = 'z%d' % self._nextTmp
                self._nextTmp += 1
                self._registers[reg.name] = reg
            if reg.name not in self._registers:
                self._registers[reg.name] = reg
            seen.add(reg.name)
            for child in reg.inputsTo():
                if child.name in seen: continue
                _recursivelyLabelUnnamedRegisters(child, seen)
        for regName,reg in self.namedRegisterItems():
            _recursivelyLabelUnnamedRegisters(reg)
        self._setupComplete = True
        # convenient return value so we can say net = FooNet().setup()
        return self

    def isParam(self,regName):
        """ Is the register with this name a parameter? """
        return self._registers[regName].role=='param'

    def isInput(self,regName):
        """ Is the register with this name an input? """
        return self._registers[regName].role=='input'

    def isOpOutput(self,regName):
        """ Is the register with this name defined as the output of some operation?? """
        return self._registers[regName].role=='operationOutput'

    def inputDict(self,**kw):
        """ Create dictionary to be passed to an eval routine, mapping
        register names to values, and adding default values for
        input() and param() registers if they are defined.
        """
        result = kw.copy()
        for regName,reg in self._registers.items():
            if not regName in result and reg.default is not None:
                result[regName] = reg.default
        return result

    def registers(self):
        """ Return a dictionary mapping register names to registers
        """
        assert self._setupComplete, 'registers() called before setup()'
        return self._registers

    def namedRegisterItems(self):
        """ Returns a list of all pairs (name,registerObject) where some
        python class/instance variable with the given name is bound to
        a Register object.  These are sorted by name to make tests
        easier.
        """
        keys = sorted(self.__dict__.keys() + self.__class__.__dict__.keys())
        vals = [self.__dict__.get(k) or self.__class__.__dict__.get(k) for k in keys]
        return filter(lambda (reg,regObj):isinstance(regObj,Register), zip(keys,vals))

    def operationSequence(self,reg, previouslyDefined=None):
        """ Traverse the expression tree to find the sequence of operations
        needed to compute the values associated with this register.
        """
        assert self._setupComplete, 'operationSequence() called before setup()'
        # pre-order traversal of the expression tree
        if previouslyDefined is None:
            previouslyDefined = set()
        buf = []
        for child in reg.inputsTo():
            if child.name not in previouslyDefined:
                buf += self.operationSequence(child, previouslyDefined)
                previouslyDefined.add(child.name)
        if reg.definedAs and (not reg.name in previouslyDefined):
            buf.append(reg.definedAs.asStringTuple())
        return buf

