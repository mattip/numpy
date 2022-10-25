"""Here we define the exported functions, types, etc... which need to be
exported through a global C pointer.

Each dictionary contains name -> index pair.

Whenever you change one index, you break the ABI (and the ABI version number
should be incremented). Whenever you add an item to one of the dict, the API
needs to be updated in both setup_common.py and by adding an appropriate
entry to cversion.txt (generate the hash via "python cversions.py").

When adding a function, make sure to use the next integer not used as an index
(in case you use an existing index or jump, the build will stop and raise an
exception, so it should hopefully not get unnoticed).

"""
from code_generators.genapi import StealRef

# index, type
multiarray_global_vars = {
    'NPY_NUMUSERTYPES':             (7, 'int'),
    'NPY_DEFAULT_ASSIGN_CASTING':   (292, 'NPY_CASTING'),
    'PyDataMem_DefaultHandler':     (306, 'PyObject*'),
}

multiarray_scalar_bool_values = {
    '_PyArrayScalar_BoolValues':    (9,)
}

# index, annotations
# please mark functions that have been checked to not need any annotations
multiarray_types_api = {
    'PyBigArray_Type':                  (1,),
    'PyArray_Type':                     (2, None, '&PyArray_Type'),
    # Internally, PyArrayDescr_Type is a PyArray_DTypeMeta,
    # the following also defines PyArrayDescr_TypeFull (Full appended)
    #'PyArrayDescr_Type':                (3, "PyArray_DTypeMeta"),
    'PyArrayDescr_Type':                (3, None, "&PyArrayDescr_Type"),
    'PyArrayFlags_Type':                (4, None, "_PyArrayFlags_Type_p"),
    'PyArrayIter_Type':                 (5, None, "_PyArrayIter_Type_p"),
    'PyArrayMultiIter_Type':            (6, None, "_PyArrayMultiIter_Type_p"),
    'PyBoolArrType_Type':               (8, None, '&PyBoolArrType_Type'),
    'PyGenericArrType_Type':            (10, None, "&PyGenericArrType_Type"),
    'PyNumberArrType_Type':             (11, None, "&PyNumberArrType_Type"),
    'PyIntegerArrType_Type':            (12, None, '&PyIntegerArrType_Type'),
    'PySignedIntegerArrType_Type':      (13, None, '&PySignedIntegerArrType_Type'),
    'PyUnsignedIntegerArrType_Type':    (14, None, '&PyUnsignedIntegerArrType_Type'),
    'PyInexactArrType_Type':            (15, None, '&PyInexactArrType_Type'),
    'PyFloatingArrType_Type':           (16, None, '&PyFloatingArrType_Type'),
    'PyComplexFloatingArrType_Type':    (17, None, '&PyComplexFloatingArrType_Type'),
    'PyFlexibleArrType_Type':           (18, None, '&PyFlexibleArrType_Type'),
    'PyCharacterArrType_Type':          (19, None, '&PyCharacterArrType_Type'),
    'PyByteArrType_Type':               (20, None, '&PyByteArrType_Type'),
    'PyShortArrType_Type':              (21, None, '&PyShortArrType_Type'),
    'PyIntArrType_Type':                (22, None, '&PyIntArrType_Type'),
    'PyLongArrType_Type':               (23, None, '&PyLongArrType_Type'),
    'PyLongLongArrType_Type':           (24, None, '&PyLongLongArrType_Type'),
    'PyUByteArrType_Type':              (25, None, '&PyUByteArrType_Type'),
    'PyUShortArrType_Type':             (26, None, '&PyUShortArrType_Type'),
    'PyUIntArrType_Type':               (27, None, '&PyUIntArrType_Type'),
    'PyULongArrType_Type':              (28, None, '&PyULongArrType_Type'),
    'PyULongLongArrType_Type':          (29, None, '&PyULongLongArrType_Type'),
    'PyFloatArrType_Type':              (30, None, '&PyFloatArrType_Type'),
    'PyDoubleArrType_Type':             (31, None, '&PyDoubleArrType_Type'),
    'PyLongDoubleArrType_Type':         (32, None, '&PyLongDoubleArrType_Type'),
    'PyCFloatArrType_Type':             (33, None, '&PyCFloatArrType_Type'),
    'PyCDoubleArrType_Type':            (34, None, '&PyCDoubleArrType_Type'),
    'PyCLongDoubleArrType_Type':        (35, None, '&PyCLongDoubleArrType_Type'),
    'PyObjectArrType_Type':             (36, None, '&PyObjectArrType_Type'),
    'PyStringArrType_Type':             (37, None, '&PyStringArrType_Type'),
    'PyUnicodeArrType_Type':            (38, None, '&PyUnicodeArrType_Type'),
    'PyVoidArrType_Type':               (39, None, '&PyVoidArrType_Type'),
    # End 1.5 API
    'PyTimeIntegerArrType_Type':        (214, None, '&PyTimeIntegerArrType_Type'),
    'PyDatetimeArrType_Type':           (215, None, '&PyDatetimeArrType_Type'),
    'PyTimedeltaArrType_Type':          (216, None, '&PyTimedeltaArrType_Type'),
    'PyHalfArrType_Type':               (217, None, '&PyHalfArrType_Type'),
    'NpyIter_Type':                     (218, None, "_NpyIter_Type_p"),
    # End 1.6 API
}

# define NPY_NUMUSERTYPES (*(int *)PyArray_API[6])
# define PyBoolArrType_Type (*(PyTypeObject *)PyArray_API[7])
# define _PyArrayScalar_BoolValues ((PyBoolScalarObject *)PyArray_API[8])

multiarray_funcs_api = {
    'PyArray_GetNDArrayCVersion':           (0,),
    'PyArray_SetNumericOps':                (40,),
    'PyArray_GetNumericOps':                (41,),
    'PyArray_INCREF':                       (42,),
    'PyArray_XDECREF':                      (43,),
    'PyArray_SetStringFunction':            (44,),
    'PyArray_DescrFromType':                (45,),
    'PyArray_TypeObjectFromType':           (46,),
    'PyArray_Zero':                         (47,),
    'PyArray_One':                          (48,),
    'PyArray_CastToType':                   (49, StealRef(2)),
    'PyArray_CastTo':                       (50,),
    'PyArray_CastAnyTo':                    (51,),
    'PyArray_CanCastSafely':                (52,),
    'PyArray_CanCastTo':                    (53,),
    'PyArray_ObjectType':                   (54,),
    'PyArray_DescrFromObject':              (55,),
    'PyArray_ConvertToCommonType':          (56,),
    'PyArray_DescrFromScalar':              (57,),
    'PyArray_DescrFromTypeObject':          (58,),
    'PyArray_Size':                         (59,),
    'PyArray_Scalar':                       (60,),
    'PyArray_FromScalar':                   (61, StealRef(2)),
    'PyArray_ScalarAsCtype':                (62,),
    'PyArray_CastScalarToCtype':            (63,),
    'PyArray_CastScalarDirect':             (64,),
    'PyArray_ScalarFromObject':             (65,),
    'PyArray_GetCastFunc':                  (66,),
    'PyArray_FromDims':                     (67,),
    'PyArray_FromDimsAndDataAndDescr':      (68, StealRef(3)),
    'PyArray_FromAny':                      (69, StealRef(2)),
    'PyArray_EnsureArray':                  (70, StealRef(1)),
    'PyArray_EnsureAnyArray':               (71, StealRef(1)),
    'PyArray_FromFile':                     (72,),
    'PyArray_FromString':                   (73,),
    'PyArray_FromBuffer':                   (74,),
    'PyArray_FromIter':                     (75, StealRef(2)),
    'PyArray_Return':                       (76, StealRef(1)),
    'PyArray_GetField':                     (77, StealRef(2)),
    'PyArray_SetField':                     (78, StealRef(2)),
    'PyArray_Byteswap':                     (79,),
    'PyArray_Resize':                       (80,),
    'PyArray_MoveInto':                     (81,),
    'PyArray_CopyInto':                     (82,),
    'PyArray_CopyAnyInto':                  (83,),
    'PyArray_CopyObject':                   (84,),
    'PyArray_NewCopy':                      (85,),
    'PyArray_ToList':                       (86,),
    'PyArray_ToString':                     (87,),
    'PyArray_ToFile':                       (88,),
    'PyArray_Dump':                         (89,),
    'PyArray_Dumps':                        (90,),
    'PyArray_ValidType':                    (91,),
    'PyArray_UpdateFlags':                  (92,),
    'PyArray_New':                          (93,),
    'PyArray_NewFromDescr':                 (94, StealRef(2)),
    'PyArray_DescrNew':                     (95,),
    'PyArray_DescrNewFromType':             (96,),
    'PyArray_GetPriority':                  (97,),
    'PyArray_IterNew':                      (98,),
    'PyArray_MultiIterNew':                 (99,),
    'PyArray_PyIntAsInt':                   (100,),
    'PyArray_PyIntAsIntp':                  (101,),
    'PyArray_Broadcast':                    (102,),
    'PyArray_FillObjectArray':              (103,),
    'PyArray_FillWithScalar':               (104,),
    'PyArray_CheckStrides':                 (105,),
    'PyArray_DescrNewByteorder':            (106,),
    'PyArray_IterAllButAxis':               (107,),
    'PyArray_CheckFromAny':                 (108, StealRef(2)),
    'PyArray_FromArray':                    (109, StealRef(2)),
    'PyArray_FromInterface':                (110,),
    'PyArray_FromStructInterface':          (111,),
    'PyArray_FromArrayAttr':                (112,),
    'PyArray_ScalarKind':                   (113,),
    'PyArray_CanCoerceScalar':              (114,),
    'PyArray_NewFlagsObject':               (115,),
    'PyArray_CanCastScalar':                (116,),
    'PyArray_CompareUCS4':                  (117,),
    'PyArray_RemoveSmallest':               (118,),
    'PyArray_ElementStrides':               (119,),
    'PyArray_Item_INCREF':                  (120,),
    'PyArray_Item_XDECREF':                 (121,),
    'PyArray_FieldNames':                   (122,),
    'PyArray_Transpose':                    (123,),
    'PyArray_TakeFrom':                     (124,),
    'PyArray_PutTo':                        (125,),
    'PyArray_PutMask':                      (126,),
    'PyArray_Repeat':                       (127,),
    'PyArray_Choose':                       (128,),
    'PyArray_Sort':                         (129,),
    'PyArray_ArgSort':                      (130,),
    'PyArray_SearchSorted':                 (131,),
    'PyArray_ArgMax':                       (132,),
    'PyArray_ArgMin':                       (133,),
    'PyArray_Reshape':                      (134,),
    'PyArray_Newshape':                     (135,),
    'PyArray_Squeeze':                      (136,),
    'PyArray_View':                         (137, StealRef(2)),
    'PyArray_SwapAxes':                     (138,),
    'PyArray_Max':                          (139,),
    'PyArray_Min':                          (140,),
    'PyArray_Ptp':                          (141,),
    'PyArray_Mean':                         (142,),
    'PyArray_Trace':                        (143,),
    'PyArray_Diagonal':                     (144,),
    'PyArray_Clip':                         (145,),
    'PyArray_Conjugate':                    (146,),
    'PyArray_Nonzero':                      (147,),
    'PyArray_Std':                          (148,),
    'PyArray_Sum':                          (149,),
    'PyArray_CumSum':                       (150,),
    'PyArray_Prod':                         (151,),
    'PyArray_CumProd':                      (152,),
    'PyArray_All':                          (153,),
    'PyArray_Any':                          (154,),
    'PyArray_Compress':                     (155,),
    'PyArray_Flatten':                      (156,),
    'PyArray_Ravel':                        (157,),
    'PyArray_MultiplyList':                 (158,),
    'PyArray_MultiplyIntList':              (159,),
    'PyArray_GetPtr':                       (160,),
    'PyArray_CompareLists':                 (161,),
    'PyArray_AsCArray':                     (162, StealRef(5)),
    'PyArray_As1D':                         (163,),
    'PyArray_As2D':                         (164,),
    'PyArray_Free':                         (165,),
    'PyArray_Converter':                    (166,),
    'PyArray_IntpFromSequence':             (167,),
    'PyArray_Concatenate':                  (168,),
    'PyArray_InnerProduct':                 (169,),
    'PyArray_MatrixProduct':                (170,),
    'PyArray_CopyAndTranspose':             (171,),
    'PyArray_Correlate':                    (172,),
    'PyArray_TypestrConvert':               (173,),
    'PyArray_DescrConverter':               (174,),
    'PyArray_DescrConverter2':              (175,),
    'PyArray_IntpConverter':                (176,),
    'PyArray_BufferConverter':              (177,),
    'PyArray_AxisConverter':                (178,),
    'PyArray_BoolConverter':                (179,),
    'PyArray_ByteorderConverter':           (180,),
    'PyArray_OrderConverter':               (181,),
    'PyArray_EquivTypes':                   (182,),
    'PyArray_Zeros':                        (183, StealRef(3)),
    'PyArray_Empty':                        (184, StealRef(3)),
    'PyArray_Where':                        (185,),
    'PyArray_Arange':                       (186,),
    'PyArray_ArangeObj':                    (187,),
    'PyArray_SortkindConverter':            (188,),
    'PyArray_LexSort':                      (189,),
    'PyArray_Round':                        (190,),
    'PyArray_EquivTypenums':                (191,),
    'PyArray_RegisterDataType':             (192,),
    'PyArray_RegisterCastFunc':             (193,),
    'PyArray_RegisterCanCast':              (194,),
    'PyArray_InitArrFuncs':                 (195,),
    'PyArray_IntTupleFromIntp':             (196,),
    'PyArray_TypeNumFromName':              (197,),
    'PyArray_ClipmodeConverter':            (198,),
    'PyArray_OutputConverter':              (199,),
    'PyArray_BroadcastToShape':             (200,),
    '_PyArray_SigintHandler':               (201,),
    '_PyArray_GetSigintBuf':                (202,),
    'PyArray_DescrAlignConverter':          (203,),
    'PyArray_DescrAlignConverter2':         (204,),
    'PyArray_SearchsideConverter':          (205,),
    'PyArray_CheckAxis':                    (206,),
    'PyArray_OverflowMultiplyList':         (207,),
    'PyArray_CompareString':                (208,),
    'PyArray_MultiIterFromObjects':         (209,),
    'PyArray_GetEndianness':                (210,),
    'PyArray_GetNDArrayCFeatureVersion':    (211,),
    'PyArray_Correlate2':                   (212,),
    'PyArray_NeighborhoodIterNew':          (213,),
    # End 1.5 API
    'PyArray_SetDatetimeParseFunction':     (219,),
    'PyArray_DatetimeToDatetimeStruct':     (220,),
    'PyArray_TimedeltaToTimedeltaStruct':   (221,),
    'PyArray_DatetimeStructToDatetime':     (222,),
    'PyArray_TimedeltaStructToTimedelta':   (223,),
    # NDIter API
    'NpyIter_New':                          (224,),
    'NpyIter_MultiNew':                     (225,),
    'NpyIter_AdvancedNew':                  (226,),
    'NpyIter_Copy':                         (227,),
    'NpyIter_Deallocate':                   (228,),
    'NpyIter_HasDelayedBufAlloc':           (229,),
    'NpyIter_HasExternalLoop':              (230,),
    'NpyIter_EnableExternalLoop':           (231,),
    'NpyIter_GetInnerStrideArray':          (232,),
    'NpyIter_GetInnerLoopSizePtr':          (233,),
    'NpyIter_Reset':                        (234,),
    'NpyIter_ResetBasePointers':            (235,),
    'NpyIter_ResetToIterIndexRange':        (236,),
    'NpyIter_GetNDim':                      (237,),
    'NpyIter_GetNOp':                       (238,),
    'NpyIter_GetIterNext':                  (239,),
    'NpyIter_GetIterSize':                  (240,),
    'NpyIter_GetIterIndexRange':            (241,),
    'NpyIter_GetIterIndex':                 (242,),
    'NpyIter_GotoIterIndex':                (243,),
    'NpyIter_HasMultiIndex':                (244,),
    'NpyIter_GetShape':                     (245,),
    'NpyIter_GetGetMultiIndex':             (246,),
    'NpyIter_GotoMultiIndex':               (247,),
    'NpyIter_RemoveMultiIndex':             (248,),
    'NpyIter_HasIndex':                     (249,),
    'NpyIter_IsBuffered':                   (250,),
    'NpyIter_IsGrowInner':                  (251,),
    'NpyIter_GetBufferSize':                (252,),
    'NpyIter_GetIndexPtr':                  (253,),
    'NpyIter_GotoIndex':                    (254,),
    'NpyIter_GetDataPtrArray':              (255,),
    'NpyIter_GetDescrArray':                (256,),
    'NpyIter_GetOperandArray':              (257,),
    'NpyIter_GetIterView':                  (258,),
    'NpyIter_GetReadFlags':                 (259,),
    'NpyIter_GetWriteFlags':                (260,),
    'NpyIter_DebugPrint':                   (261,),
    'NpyIter_IterationNeedsAPI':            (262,),
    'NpyIter_GetInnerFixedStrideArray':     (263,),
    'NpyIter_RemoveAxis':                   (264,),
    'NpyIter_GetAxisStrideArray':           (265,),
    'NpyIter_RequiresBuffering':            (266,),
    'NpyIter_GetInitialDataPtrArray':       (267,),
    'NpyIter_CreateCompatibleStrides':      (268,),
    #
    'PyArray_CastingConverter':             (269,),
    'PyArray_CountNonzero':                 (270,),
    'PyArray_PromoteTypes':                 (271,),
    'PyArray_MinScalarType':                (272,),
    'PyArray_ResultType':                   (273,),
    'PyArray_CanCastArrayTo':               (274,),
    'PyArray_CanCastTypeTo':                (275,),
    'PyArray_EinsteinSum':                  (276,),
    'PyArray_NewLikeArray':                 (277, StealRef(3)),
    'PyArray_GetArrayParamsFromObject':     (278,),
    'PyArray_ConvertClipmodeSequence':      (279,),
    'PyArray_MatrixProduct2':               (280,),
    # End 1.6 API
    'NpyIter_IsFirstVisit':                 (281,),
    'PyArray_SetBaseObject':                (282, StealRef(2)),
    'PyArray_CreateSortedStridePerm':       (283,),
    'PyArray_RemoveAxesInPlace':            (284,),
    'PyArray_DebugPrint':                   (285,),
    'PyArray_FailUnlessWriteable':          (286,),
    'PyArray_SetUpdateIfCopyBase':          (287, StealRef(2)),
    'PyDataMem_NEW':                        (288,),
    'PyDataMem_FREE':                       (289,),
    'PyDataMem_RENEW':                      (290,),
    'PyDataMem_SetEventHook':               (291,),
    'PyArray_MapIterSwapAxes':              (293,),
    'PyArray_MapIterArray':                 (294,),
    'PyArray_MapIterNext':                  (295,),
    # End 1.7 API
    'PyArray_Partition':                    (296,),
    'PyArray_ArgPartition':                 (297,),
    'PyArray_SelectkindConverter':          (298,),
    'PyDataMem_NEW_ZEROED':                 (299,),
    # End 1.8 API
    # End 1.9 API
    'PyArray_CheckAnyScalarExact':          (300,),
    # End 1.10 API
    'PyArray_MapIterArrayCopyIfOverlap':    (301,),
    # End 1.13 API
    'PyArray_ResolveWritebackIfCopy':       (302,),
    'PyArray_SetWritebackIfCopyBase':       (303,),
    # End 1.14 API
    'PyDataMem_SetHandler':                 (304,),
    'PyDataMem_GetHandler':                 (305,),
    # End 1.21 API
}

# HPy API:

# index, type
hpy_multiarray_global_vars = {
    # 'NPY_NUMUSERTYPES':             (7, 'int'),
    # 'NPY_DEFAULT_ASSIGN_CASTING':   (292, 'NPY_CASTING'),
    # 'HPyDataMem_DefaultHandler':     (306, 'HPy'),
}

hpy_multiarray_scalar_bool_values = {
    '_HPyArrayScalar_BoolValues':    (9,)
}

# index, annotations
# please mark functions that have been checked to not need any annotations
hpy_multiarray_global_types_api = {
    # 'HPyBigArray_Type':                  (1,),
    'HPyArray_Type':                     (2,),
    'HPyArrayDescr_Type':                (3,),
    # 'HPyArrayFlags_Type':                (4,),
    # 'HPyArrayIter_Type':                 (5,),
    # 'HPyArrayMultiIter_Type':            (6,),
    'HPyBoolArrType_Type':               (8,),
    'HPyGenericArrType_Type':            (10,),
    'HPyNumberArrType_Type':             (11,),
    'HPyIntegerArrType_Type':            (12,),
    'HPySignedIntegerArrType_Type':      (13,),
    'HPyUnsignedIntegerArrType_Type':    (14,),
    'HPyInexactArrType_Type':            (15,),
    'HPyFloatingArrType_Type':           (16,),
    'HPyComplexFloatingArrType_Type':    (17,),
    'HPyFlexibleArrType_Type':           (18,),
    'HPyCharacterArrType_Type':          (19,),
    'HPyByteArrType_Type':               (20,),
    'HPyShortArrType_Type':              (21,),
    'HPyIntArrType_Type':                (22,),
    'HPyLongArrType_Type':               (23,),
    'HPyLongLongArrType_Type':           (24,),
    'HPyUByteArrType_Type':              (25,),
    'HPyUShortArrType_Type':             (26,),
    'HPyUIntArrType_Type':               (27,),
    'HPyULongArrType_Type':              (28,),
    'HPyULongLongArrType_Type':          (29,),
    'HPyFloatArrType_Type':              (30,),
    'HPyDoubleArrType_Type':             (31,),
    'HPyLongDoubleArrType_Type':         (32,),
    'HPyCFloatArrType_Type':             (33,),
    'HPyCDoubleArrType_Type':            (34,),
    'HPyCLongDoubleArrType_Type':        (35,),
    'HPyObjectArrType_Type':             (36,),
    'HPyStringArrType_Type':             (37,),
    'HPyUnicodeArrType_Type':            (38,),
    'HPyVoidArrType_Type':               (39,),
    # End 1.5 API
    # 'HPyTimeIntegerArrType_Type':        (214,),
    'HPyDatetimeArrType_Type':           (215,),
    'HPyTimedeltaArrType_Type':          (216,),
    'HPyHalfArrType_Type':               (217,),
    # 'HNpyIter_Type':                     (218,),
    # End 1.6 API
}

HPY_API_START = 0
hpy_multiarray_funcs_api = {
    'HPyArray_GetNDArrayCVersion':           (0,),
    'HPyArray_SetNumericOps':                (40,),
    'HPyArray_GetNumericOps':                (41,),
    # 'HPyArray_INCREF':                       (42,),
    # 'HPyArray_XDECREF':                      (43,),
    # 'HPyArray_SetStringFunction':            (44,),
    'HPyArray_DescrFromType':                (45,),
    # 'HPyArray_TypeObjectFromType':           (46,),
    # 'HPyArray_Zero':                         (47,),
    # 'HPyArray_One':                          (48,),
    'HPyArray_CastToType':                   (49,),
    # 'HPyArray_CastTo':                       (50,),
    # 'HPyArray_CastAnyTo':                    (51,),
    # 'HPyArray_CanCastSafely':                (52,),
    # 'HPyArray_CanCastTo':                    (53,),
    'HPyArray_ObjectType':                   (54,),
    'HPyArray_DescrFromObject':              (55,),
    # 'HPyArray_ConvertToCommonType':          (56,),
    'HPyArray_DescrFromScalar':              (57,),
    'HPyArray_DescrFromTypeObject':          (58,),
    'HPyArray_Size':                         (59,),
    'HPyArray_Scalar':                       (60,),
    'HPyArray_FromScalar':                   (61,),
    # 'HPyArray_ScalarAsCtype':                (62,),
    # 'HPyArray_CastScalarToCtype':            (63,),
    # 'HPyArray_CastScalarDirect':             (64,),
    # 'HPyArray_ScalarFromObject':             (65,),
    'HPyArray_GetCastFunc':                  (66,),
    # 'HPyArray_FromDims':                     (67,),
    # 'HPyArray_FromDimsAndDataAndDescr':      (68,),
    'HPyArray_FromAny':                      (69,),
    'HPyArray_EnsureArray':                  (70,),
    'HPyArray_EnsureAnyArray':               (71,),
    'HPyArray_FromFile':                     (72,),
    'HPyArray_FromString':                   (73,),
    'HPyArray_FromBuffer':                   (74,),
    'HPyArray_FromIter':                     (75,),
    'HPyArray_Return':                       (76,),
    # 'HPyArray_GetField':                     (77,),
    # 'HPyArray_SetField':                     (78,),
    # 'HPyArray_Byteswap':                     (79,),
    # 'HPyArray_Resize':                       (80,),
    # 'HPyArray_MoveInto':                     (81,),
    'HPyArray_CopyInto':                     (82,),
    'HPyArray_CopyAnyInto':                  (83,),
    'HPyArray_CopyObject':                   (84,),
    'HPyArray_NewCopy':                      (85,),
    # 'HPyArray_ToList':                       (86,),
    # 'HPyArray_ToString':                     (87,),
    # 'HPyArray_ToFile':                       (88,),
    # 'HPyArray_Dump':                         (89,),
    # 'HPyArray_Dumps':                        (90,),
    # 'HPyArray_ValidType':                    (91,),
    'HPyArray_UpdateFlags':                  (92,),
    'HPyArray_New':                          (93,),
    'HPyArray_NewFromDescr':                 (94,),
    'HPyArray_DescrNew':                     (95,),
    'HPyArray_DescrNewFromType':             (96,),
    'HPyArray_GetPriority':                  (97,),
    # 'HPyArray_IterNew':                      (98,),
    # 'HPyArray_MultiIterNew':                 (99,),
    'HPyArray_PyIntAsInt':                   (100,),
    'HPyArray_PyIntAsIntp':                  (101,),
    # 'HPyArray_Broadcast':                    (102,),
    'HPyArray_FillObjectArray':              (103,),
    # 'HPyArray_FillWithScalar':               (104,),
    # 'HPyArray_CheckStrides':                 (105,),
    'HPyArray_DescrNewByteorder':            (106,),
    'HPyArray_IterAllButAxis':               (107,),
    'HPyArray_CheckFromAny':                 (108,),
    'HPyArray_FromArray':                    (109,),
    'HPyArray_FromInterface':                (110,),
    'HPyArray_FromStructInterface':          (111,),
    # 'HPyArray_FromArrayAttr':                (112,),
    # 'HPyArray_ScalarKind':                   (113,),
    # 'HPyArray_CanCoerceScalar':              (114,),
    'HPyArray_NewFlagsObject':               (115,),
    # 'HPyArray_CanCastScalar':                (116,),
    # 'HPyArray_CompareUCS4':                  (117,),
    # 'HPyArray_RemoveSmallest':               (118,),
    'HPyArray_ElementStrides':               (119,),
    'HPyArray_Item_INCREF':                  (120,),
    'HPyArray_Item_XDECREF':                 (121,),
    # 'HPyArray_FieldNames':                   (122,),
    'HPyArray_Transpose':                    (123,),
    'HPyArray_TakeFrom':                     (124,),
    # 'HPyArray_PutTo':                        (125,),
    'HPyArray_PutMask':                      (126,),
    # 'HPyArray_Repeat':                       (127,),
    # 'HPyArray_Choose':                       (128,),
    # 'HPyArray_Sort':                         (129,),
    # 'HPyArray_ArgSort':                      (130,),
    # 'HPyArray_SearchSorted':                 (131,),
    # 'HPyArray_ArgMax':                       (132,),
    # 'HPyArray_ArgMin':                       (133,),
    # 'HPyArray_Reshape':                      (134,),
    'HPyArray_Newshape':                     (135,),
    # 'HPyArray_Squeeze':                      (136,),
    'HPyArray_View':                         (137,),
    # 'HPyArray_SwapAxes':                     (138,),
    # 'HPyArray_Max':                          (139,),
    # 'HPyArray_Min':                          (140,),
    # 'HPyArray_Ptp':                          (141,),
    # 'HPyArray_Mean':                         (142,),
    # 'HPyArray_Trace':                        (143,),
    # 'HPyArray_Diagonal':                     (144,),
    # 'HPyArray_Clip':                         (145,),
    # 'HPyArray_Conjugate':                    (146,),
    'HPyArray_Nonzero':                      (147,),
    # 'HPyArray_Std':                          (148,),
    # 'HPyArray_Sum':                          (149,),
    # 'HPyArray_CumSum':                       (150,),
    # 'HPyArray_Prod':                         (151,),
    # 'HPyArray_CumProd':                      (152,),
    # 'HPyArray_All':                          (153,),
    # 'HPyArray_Any':                          (154,),
    'HPyArray_Compress':                     (155,),
    'HPyArray_Flatten':                      (156,),
    'HPyArray_Ravel':                        (157,),
    # 'HPyArray_MultiplyList':                 (158,),
    # 'HPyArray_MultiplyIntList':              (159,),
    # 'HPyArray_GetPtr':                       (160,),
    # 'HPyArray_CompareLists':                 (161,),
    # 'HPyArray_AsCArray':                     (162,),
    # 'HPyArray_As1D':                         (163,),
    # 'HPyArray_As2D':                         (164,),
    # 'HPyArray_Free':                         (165,),
    'HPyArray_Converter':                    (166,),
    # 'HPyArray_IntpFromSequence':             (167,),
    'HPyArray_Concatenate':                  (168,),
    'HPyArray_InnerProduct':                 (169,),
    'HPyArray_MatrixProduct':                (170,),
    'HPyArray_CopyAndTranspose':             (171,),
    # 'HPyArray_Correlate':                    (172,),
    # 'HPyArray_TypestrConvert':               (173,),
    'HPyArray_DescrConverter':               (174,),
    'HPyArray_DescrConverter2':              (175,),
    'HPyArray_IntpConverter':                (176,),
    'HPyArray_BufferConverter':              (177,),
    'HPyArray_AxisConverter':                (178,),
    'HPyArray_BoolConverter':                (179,),
    # 'HPyArray_ByteorderConverter':           (180,),
    'HPyArray_OrderConverter':               (181,),
    'HPyArray_EquivTypes':                   (182,),
    'HPyArray_Zeros':                        (183,),
    'HPyArray_Empty':                        (184,),
    'HPyArray_Where':                        (185,),
    # 'HPyArray_Arange':                       (186,),
    'HPyArray_ArangeObj':                    (187,),
    # 'HPyArray_SortkindConverter':            (188,),
    # 'HPyArray_LexSort':                      (189,),
    # 'HPyArray_Round':                        (190,),
    # 'HPyArray_EquivTypenums':                (191,),
    'HPyArray_RegisterDataType':             (192,),
    'HPyArray_RegisterCastFunc':             (193,),
    # 'HPyArray_RegisterCanCast':              (194,),
    # 'HPyArray_InitArrFuncs':                 (195,),
    'HPyArray_IntTupleFromIntp':             (196,),
    # 'HPyArray_TypeNumFromName':              (197,),
    'HPyArray_ClipmodeConverter':            (198,),
    'HPyArray_OutputConverter':              (199,),
    # 'HPyArray_BroadcastToShape':             (200,),
    # '_HPyArray_SigintHandler':               (201,),
    # '_HPyArray_GetSigintBuf':                (202,),
    # 'HPyArray_DescrAlignConverter':          (203,),
    # 'HPyArray_DescrAlignConverter2':         (204,),
    # 'HPyArray_SearchsideConverter':          (205,),
    'HPyArray_CheckAxis':                    (206,),
    # 'HPyArray_OverflowMultiplyList':         (207,),
    # 'HPyArray_CompareString':                (208,),
    # 'HPyArray_MultiIterFromObjects':         (209,),
    # 'HPyArray_GetEndianness':                (210,),
    # 'HPyArray_GetNDArrayCFeatureVersion':    (211,),
    # 'HPyArray_Correlate2':                   (212,),
    # 'HPyArray_NeighborhoodIterNew':          (213,),
    # End 1.5 API
    # 'HPyArray_SetDatetimeParseFunction':     (219,),
    # 'HPyArray_DatetimeToDatetimeStruct':     (220,),
    # 'HPyArray_TimedeltaToTimedeltaStruct':   (221,),
    # 'HPyArray_DatetimeStructToDatetime':     (222,),
    # 'HPyArray_TimedeltaStructToTimedelta':   (223,),
    # NDIter API
    'HNpyIter_New':                          (224,),
    'HNpyIter_MultiNew':                     (225,),
    'HNpyIter_AdvancedNew':                  (226,),
    # 'HNpyIter_Copy':                         (227,),
    'HNpyIter_Deallocate':                   (228,),
    # 'HNpyIter_HasDelayedBufAlloc':           (229,),
    # 'HNpyIter_HasExternalLoop':              (230,),
    'HNpyIter_EnableExternalLoop':           (231,),
    # 'HNpyIter_GetInnerStrideArray':          (232,),
    # 'HNpyIter_GetInnerLoopSizePtr':          (233,),
    'HNpyIter_Reset':                        (234,),
    'HNpyIter_ResetBasePointers':            (235,),
    # 'HNpyIter_ResetToIterIndexRange':        (236,),
    # 'HNpyIter_GetNDim':                      (237,),
    # 'HNpyIter_GetNOp':                       (238,),
    'HNpyIter_GetIterNext':                  (239,),
    # 'HNpyIter_GetIterSize':                  (240,),
    # 'HNpyIter_GetIterIndexRange':            (241,),
    # 'HNpyIter_GetIterIndex':                 (242,),
    'HNpyIter_GotoIterIndex':                (243,),
    # 'HNpyIter_HasMultiIndex':                (244,),
    # 'HNpyIter_GetShape':                     (245,),
    'HNpyIter_GetGetMultiIndex':             (246,),
    # 'HNpyIter_GotoMultiIndex':               (247,),
    'HNpyIter_RemoveMultiIndex':             (248,),
    # 'HNpyIter_HasIndex':                     (249,),
    # 'HNpyIter_IsBuffered':                   (250,),
    # 'HNpyIter_IsGrowInner':                  (251,),
    # 'HNpyIter_GetBufferSize':                (252,),
    # 'HNpyIter_GetIndexPtr':                  (253,),
    # 'HNpyIter_GotoIndex':                    (254,),
    # 'HNpyIter_GetDataPtrArray':              (255,),
    'HNpyIter_GetDescrArray':                (256,),
    'HNpyIter_GetOperandArray':              (257,),
    'HNpyIter_GetIterView':                  (258,),
    # 'HNpyIter_GetReadFlags':                 (259,),
    # 'HNpyIter_GetWriteFlags':                (260,),
    'HNpyIter_DebugPrint':                   (261,),
    # 'HNpyIter_IterationNeedsAPI':            (262,),
    'HNpyIter_GetInnerFixedStrideArray':     (263,),
    'HNpyIter_RemoveAxis':                   (264,),
    # 'HNpyIter_GetAxisStrideArray':           (265,),
    # 'HNpyIter_RequiresBuffering':            (266,),
    # 'HNpyIter_GetInitialDataPtrArray':       (267,),
    'HNpyIter_CreateCompatibleStrides':      (268,),
    #
    'HPyArray_CastingConverter':             (269,),
    # 'HPyArray_CountNonzero':                 (270,),
    'HPyArray_PromoteTypes':                 (271,),
    'HPyArray_MinScalarType':                (272,),
    'HPyArray_ResultType':                   (273,),
    'HPyArray_CanCastArrayTo':               (274,),
    'HPyArray_CanCastTypeTo':                (275,),
    # 'HPyArray_EinsteinSum':                  (276,),
    'HPyArray_NewLikeArray':                 (277,),
    # 'HPyArray_GetArrayParamsFromObject':     (278,),
    'HPyArray_ConvertClipmodeSequence':      (279,),
    'HPyArray_MatrixProduct2':               (280,),
    # End 1.6 API
    # 'HNpyIter_IsFirstVisit':                 (281,),
    'HPyArray_SetBaseObject':                (282,),
    # 'HPyArray_CreateSortedStridePerm':       (283,),
    # 'HPyArray_RemoveAxesInPlace':            (284,),
    # 'HPyArray_DebugPrint':                   (285,),
    'HPyArray_FailUnlessWriteable':          (286,),
    # 'HPyArray_SetUpdateIfCopyBase':          (287,),
    # 'HPyDataMem_NEW':                        (288,),
    # 'HPyDataMem_FREE':                       (289,),
    # 'HPyDataMem_RENEW':                      (290,),
    # 'HPyDataMem_SetEventHook':               (291,),
    'HPyArray_MapIterSwapAxes':              (293,),
    # 'HPyArray_MapIterArray':                 (294,),
    # 'HPyArray_MapIterNext':                  (295,),
    # End 1.7 API
    # 'HPyArray_Partition':                    (296,),
    # 'HPyArray_ArgPartition':                 (297,),
    # 'HPyArray_SelectkindConverter':          (298,),
    # 'HPyDataMem_NEW_ZEROED':                 (299,),
    # End 1.8 API
    # End 1.9 API
    'HPyArray_CheckAnyScalarExact':          (300,),
    # End 1.10 API
    'HPyArray_MapIterArrayCopyIfOverlap':    (301,),
    # End 1.13 API
    'HPyArray_ResolveWritebackIfCopy':       (302,),
    'HPyArray_SetWritebackIfCopyBase':       (303,),
    # End 1.14 API
    'HPyDataMem_SetHandler':                 (304,),
    'HPyDataMem_GetHandler':                 (305,),
    # End 1.21 API    
}

ufunc_types_api = {
    'PyUFunc_Type':                             (0, None, '&PyUFunc_Type')
}

ufunc_funcs_api = {
    'PyUFunc_FromFuncAndData':                  (1,),
    'PyUFunc_RegisterLoopForType':              (2,),
    'PyUFunc_GenericFunction':                  (3,),
    'PyUFunc_f_f_As_d_d':                       (4,),
    'PyUFunc_d_d':                              (5,),
    'PyUFunc_f_f':                              (6,),
    'PyUFunc_g_g':                              (7,),
    'PyUFunc_F_F_As_D_D':                       (8,),
    'PyUFunc_F_F':                              (9,),
    'PyUFunc_D_D':                              (10,),
    'PyUFunc_G_G':                              (11,),
    'PyUFunc_O_O':                              (12,),
    'PyUFunc_ff_f_As_dd_d':                     (13,),
    'PyUFunc_ff_f':                             (14,),
    'PyUFunc_dd_d':                             (15,),
    'PyUFunc_gg_g':                             (16,),
    'PyUFunc_FF_F_As_DD_D':                     (17,),
    'PyUFunc_DD_D':                             (18,),
    'PyUFunc_FF_F':                             (19,),
    'PyUFunc_GG_G':                             (20,),
    'PyUFunc_OO_O':                             (21,),
    'PyUFunc_O_O_method':                       (22,),
    'PyUFunc_OO_O_method':                      (23,),
    'PyUFunc_On_Om':                            (24,),
    'PyUFunc_GetPyValues':                      (25,),
    'PyUFunc_checkfperr':                       (26,),
    'PyUFunc_clearfperr':                       (27,),
    'PyUFunc_getfperr':                         (28,),
    'PyUFunc_handlefperr':                      (29,),
    'PyUFunc_ReplaceLoopBySignature':           (30,),
    'PyUFunc_FromFuncAndDataAndSignature':      (31,),
    'PyUFunc_SetUsesArraysAsData':              (32,),
    # End 1.5 API
    'PyUFunc_e_e':                              (33,),
    'PyUFunc_e_e_As_f_f':                       (34,),
    'PyUFunc_e_e_As_d_d':                       (35,),
    'PyUFunc_ee_e':                             (36,),
    'PyUFunc_ee_e_As_ff_f':                     (37,),
    'PyUFunc_ee_e_As_dd_d':                     (38,),
    # End 1.6 API
    'PyUFunc_DefaultTypeResolver':              (39,),
    'PyUFunc_ValidateCasting':                  (40,),
    # End 1.7 API
    'PyUFunc_RegisterLoopForDescr':             (41,),
    # End 1.8 API
    'PyUFunc_FromFuncAndDataAndSignatureAndIdentity': (42,),
    # End 1.16 API
}

# List of all the dicts which define the C API
# XXX: DO NOT CHANGE THE ORDER OF TUPLES BELOW !
multiarray_api = (
        multiarray_global_vars,
        multiarray_scalar_bool_values,
        multiarray_types_api,
        multiarray_funcs_api,
)

hpy_multiarray_api = (
        hpy_multiarray_global_vars,
        hpy_multiarray_scalar_bool_values,
        hpy_multiarray_global_types_api,
        hpy_multiarray_funcs_api,
)

ufunc_api = (
        ufunc_funcs_api,
        ufunc_types_api
)

full_api = multiarray_api + ufunc_api
