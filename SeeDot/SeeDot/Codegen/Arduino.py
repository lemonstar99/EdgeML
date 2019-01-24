# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

from Codegen.CodegenBase import CodegenBase

import IR.IR as IR
import IR.IRUtil as IRUtil

import Type
from Util import *

class Arduino(CodegenBase):

	def __init__(self, writer, decls, expts, intvs, cnsts, expTables, globalVars):
		self.out = writer
		self.decls = decls
		self.expts = expts
		self.intvs = intvs
		self.cnsts = cnsts
		self.expTables = expTables
		self.globalVars = globalVars

	def printPrefix(self):
		self.printArduinoIncludes()

		self.printExpTables()
		
		self.printArduinoHeader()

		self.printVarDecls()

		self.printConstDecls()
		
		self.out.printf('\n')

	def printArduinoIncludes(self):
		self.out.printf('#include <Arduino.h>\n\n', indent=True)
		self.out.printf('#include "config.h"\n', indent=True)
		self.out.printf('#include "predict.h"\n', indent=True)
		self.out.printf('#include "library.h"\n', indent=True)
		self.out.printf('#include "model.h"\n\n', indent=True)
		self.out.printf('using namespace model;\n\n', indent=True)

	def printExpTables(self):
		for exp, [table, [tableVarA, tableVarB]] in self.expTables.items():
			self.printExpTable(table[0], tableVarA)
			self.printExpTable(table[1], tableVarB)
			self.out.printf('\n')

	def printExpTable(self, table_row, var):
		self.out.printf('const PROGMEM MYINT %s[%d] = {\n' % (var.idf, len(table_row)), indent = True)
		self.out.increaseIndent()
		self.out.printf('', indent = True)
		for i in range(len(table_row)):
			self.out.printf('%d, ' % table_row[i])
		self.out.decreaseIndent()
		self.out.printf('\n};\n')

	def printArduinoHeader(self):
		self.out.printf('int predict() {\n', indent=True)
		self.out.increaseIndent()

	def printSuffix(self, expr:IR.Expr):
		self.out.printf('\n')

		type = self.decls[expr.idf]

		if Type.isInt(type):
			self.out.printf('return ', indent = True)
			self.print(expr)
			self.out.printf(';\n')
		elif Type.isTensor(type):
			idfr = expr.idf
			exponent = self.expts[expr.idf]
			num = 2 ** exponent

			if type.dim == 0:
				self.out.printf('Serial.println(', indent = True)
				self.out.printf('float(' + idfr + ')*' + str(num))
				self.out.printf(', 6);\n')
			else:
				iters = []
				for i in range(type.dim):
					s = chr(ord('i') + i)
					tempVar = IR.Var(s)
					iters.append(tempVar)
				expr_1 = IRUtil.addIndex(expr, iters)
				cmds = IRUtil.loop(type.shape, iters, [IR.PrintAsFloat(expr_1, exponent)])
				self.print(IR.Prog(cmds))
		else:
			assert False

		self.out.decreaseIndent()
		self.out.printf('}\n', indent=True)

	def printVar(self, ir):
		if ir.inputVar:
			if Common.wordLength == 16:
				self.out.printf('((MYINT) pgm_read_word_near(&')
			elif Common.wordLength == 32:
				self.out.printf('((MYINT) pgm_read_dword_near(&')
			else:
				assert False
		self.out.printf('%s', ir.idf)
		for e in ir.idx:
			self.out.printf('[')
			self.print(e)
			self.out.printf(']')
		if ir.inputVar:
			self.out.printf('))')

	def printAssn(self, ir):
		if isinstance(ir.e, IR.Var) and ir.e.idf == "X":
			self.out.printf("", indent=True)
			self.print(ir.var)
			self.out.printf(" = getIntFeature(i0);\n")
		else:
			super().printAssn(ir)

	def printPrint(self, ir):
		self.out.printf('Serial.println(', indent=True)
		self.print(ir.expr)
		self.out.printf(');\n')

	def printPrintAsFloat(self, ir):
		self.out.printf('Serial.println(float(', indent=True)
		self.print(ir.expr)
		self.out.printf(') * ' + str(2 ** ir.expnt) + ', 6);')