/*
 * Copyright (c) 2008 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef ATTRIBUTE_HELPER_H
#define ATTRIBUTE_HELPER_H

#include "abort.h"
#include "attribute-accessor-helper.h"
#include "attribute.h"

#include <sstream>

/**
 * @file
 * @ingroup attributehelper
 * Attribute helper (\c ATTRIBUTE_ )macros definition.
 */

namespace ns3
{

/**
 * @ingroup attributes
 * @defgroup attributehelper Attribute Helper
 *
 * All these macros can be used to generate automatically the code
 * for subclasses of AttributeValue, AttributeAccessor, and, AttributeChecker,
 * which can be used to give attribute powers to a normal class. i.e.,
 * the user class can then effectively be made an attribute.
 *
 * There are two kinds of helper macros:
 *   -# The simple macros.
 *     - ATTRIBUTE_HELPER_HEADER(type)
 *     - ATTRIBUTE_HELPER_CPP(type)
 *   -# The more complex macros.
 *
 * The simple macros are implemented in terms of the complex
 * macros and should generally be preferred over the complex macros.
 *
 * @note
 * Because these macros generate class and function definitions, it's
 * difficult to document the results directly.  Instead, we use a
 * set of functions in print-introspected-doxygen.cc to generate
 * most of the APi documentation.  When using these macros,
 * please add the required function calls to print-introspected-doxygen.cc
 * so your new API is documented.
 */

/**
 * @ingroup attributehelper
 * @defgroup attributeimpl Attribute Implementation
 *
 * These are the internal implementation functions for the Attribute
 * system.
 *
 * Module code shouldn't need to call these directly.  Instead,
 * see \ref attributehelper.
 *
 * There are three versions of DoMakeAccessorHelperOne:
 *  - With a member variable: DoMakeAccessorHelperOne(U T::*)
 *  - With a class get functor: DoMakeAccessorHelperOne(U(T::*)() const)
 *  - With a class set method:  DoMakeAccessorHelperOne(void(T::*)(U))
 *
 * There are two pairs of DoMakeAccessorHelperTwo (four total):
 *  - Taking two arguments, a set method and get functor, in either order,
 *  - With set methods returning \c void or \c bool.
 */

/**
 * @ingroup attributeimpl
 *
 * A simple string-based attribute checker
 *
 * @tparam T    \explicit The specific AttributeValue type used to represent
 *              the Attribute.
 * @tparam BASE \explicit The AttributeChecker type corresponding to \pname{T}.
 * @param [in] name  The name of the AttributeValue type, essentially the
 *              string form of \pname{T}.
 * @param [in] underlying Underlying type name.
 * @return Ptr to AttributeChecker.
 */
template <typename T, typename BASE>
Ptr<AttributeChecker>
MakeSimpleAttributeChecker(std::string name, std::string underlying)
{
    /**
     * String-based AttributeChecker implementation.
     * @extends AttributeChecker
     */
    struct SimpleAttributeChecker : public BASE
    {
        bool Check(const AttributeValue& value) const override
        {
            return dynamic_cast<const T*>(&value) != nullptr;
        }

        std::string GetValueTypeName() const override
        {
            if (m_type.rfind("ns3::", 0) == 0)
            {
                // m_type already starts with "ns3::"
                return m_type;
            }
            return "ns3::" + m_type;
        }

        bool HasUnderlyingTypeInformation() const override
        {
            return true;
        }

        std::string GetUnderlyingTypeInformation() const override
        {
            return m_underlying;
        }

        Ptr<AttributeValue> Create() const override
        {
            return ns3::Create<T>();
        }

        bool Copy(const AttributeValue& source, AttributeValue& destination) const override
        {
            const T* src = dynamic_cast<const T*>(&source);
            T* dst = dynamic_cast<T*>(&destination);
            if (src == nullptr || dst == nullptr)
            {
                return false;
            }
            *dst = *src;
            return true;
        }

        std::string m_type;       // The name of the AttributeValue type.
        std::string m_underlying; // The underlying attribute type name.
    }* checker = new SimpleAttributeChecker();

    checker->m_type = name;
    checker->m_underlying = underlying;
    return Ptr<AttributeChecker>(checker, false);
}

} // namespace ns3

/**
 * @ingroup attributehelper
 *
 * Define the attribute accessor functions \c MakeTypeAccessor
 * for class \pname{type}.
 *
 * @param [in] type The name of the class
 *
 * This macro defines and generates the code for the implementation
 * of the \c MakeTypeAccessor template functions. This macro is typically
 * invoked in a class header to allow users of this class to view and
 * use the template functions defined here. This macro is implemented
 * through the helper templates functions ns3::MakeAccessorHelper<>.
 */
#define ATTRIBUTE_ACCESSOR_DEFINE(type)                                                            \
    template <typename T1>                                                                         \
    Ptr<const AttributeAccessor> Make##type##Accessor(T1 a1)                                       \
    {                                                                                              \
        return MakeAccessorHelper<type##Value>(a1);                                                \
    }                                                                                              \
    template <typename T1, typename T2>                                                            \
    Ptr<const AttributeAccessor> Make##type##Accessor(T1 a1, T2 a2)                                \
    {                                                                                              \
        return MakeAccessorHelper<type##Value>(a1, a2);                                            \
    }

/**
 * @ingroup attributehelper
 *
 * Declare the attribute value class \pname{nameValue}
 * for underlying class \pname{type}.
 *
 * @param [in] type The underlying type name token
 * @param [in] name The token to use in defining the accessor name.
 *
 * This macro declares the class \c typeValue associated with class \c type.
 * This macro is typically invoked in the class header file.
 *
 * This can be used directly for things like plain old data,
 * such as \c std::string, to create the attribute value class
 * StringValue, as in
 *   `ATTRIBUTE_VALUE_DEFINE_WITH_NAME(std::string, String);`
 */
#define ATTRIBUTE_VALUE_DEFINE_WITH_NAME(type, name)                                               \
    class name##Value : public AttributeValue                                                      \
    {                                                                                              \
      public:                                                                                      \
        name##Value() = default;                                                                   \
        name##Value(const type& value);                                                            \
        void Set(const type& value);                                                               \
        type Get() const;                                                                          \
        template <typename T>                                                                      \
        bool GetAccessor(T& value) const                                                           \
        {                                                                                          \
            value = T(m_value);                                                                    \
            return true;                                                                           \
        }                                                                                          \
        Ptr<AttributeValue> Copy() const override;                                                 \
        std::string SerializeToString(Ptr<const AttributeChecker> checker) const override;         \
        bool DeserializeFromString(std::string value,                                              \
                                   Ptr<const AttributeChecker> checker) override;                  \
                                                                                                   \
      private:                                                                                     \
        type m_value;                                                                              \
    }

/**
 * @ingroup attributehelper
 *
 * Declare the attribute value class \pname{nameValue}
 * for the class \pname{name}
 *
 * @param [in] name The name of the class.
 *
 * This macro declares the class \c nameValue associated to class \c name.
 * This macro is typically invoked in the class header file.
 */
#define ATTRIBUTE_VALUE_DEFINE(name) ATTRIBUTE_VALUE_DEFINE_WITH_NAME(name, name)

/**
 * @ingroup attributehelper
 *
 * Define the conversion operators class \pname{type} and
 * Attribute instances.
 *
 * @param [in] type The name of the class
 *
 * This macro defines the conversion operators for class \c type to and
 * from instances of \c typeAttribute.
 * Typically invoked in the class header file.
 *
 * @internal
 * This appears to be unused in the current code base.
 */
#define ATTRIBUTE_CONVERTER_DEFINE(type)

/**
 * @ingroup attributehelper
 *
 * Declare the AttributeChecker class \pname{typeChecker}
 * and the \c MaketypeChecker function for class \pname{type}.
 *
 * @param [in] type The name of the class
 *
 * This macro declares the \pname{typeChecker} class and the associated
 * \c MaketypeChecker function.
 *
 * (Note that the \pname{typeChecker} class needs no implementation
 * since it just inherits all its implementation from AttributeChecker.)
 *
 * Typically invoked in the class header file.
 */
#define ATTRIBUTE_CHECKER_DEFINE(type)                                                             \
    class type##Checker : public AttributeChecker                                                  \
    {                                                                                              \
    };                                                                                             \
    Ptr<const AttributeChecker> Make##type##Checker()

/**
 * @ingroup attributehelper
 *
 * Define the class methods belonging to
 * the attribute value class \pname{nameValue}
 * of the underlying class \pname{type}.
 *
 * @param [in] type The underlying type name
 * @param [in] name The token to use in defining the accessor name.
 *
 * This macro implements the \pname{typeValue} class methods
 * (including the \pname{typeValue}%::%SerializeToString
 * and \pname{typeValue}%::%DeserializeFromString methods).
 *
 * Typically invoked in the source file
 */
#define ATTRIBUTE_VALUE_IMPLEMENT_WITH_NAME(type, name)                                            \
    name##Value::name##Value(const type& value)                                                    \
        : m_value(value)                                                                           \
    {                                                                                              \
    }                                                                                              \
    void name##Value::Set(const type& v)                                                           \
    {                                                                                              \
        m_value = v;                                                                               \
    }                                                                                              \
    type name##Value::Get() const                                                                  \
    {                                                                                              \
        return m_value;                                                                            \
    }                                                                                              \
    Ptr<AttributeValue> name##Value::Copy() const                                                  \
    {                                                                                              \
        return ns3::Create<name##Value>(*this);                                                    \
    }                                                                                              \
    std::string name##Value::SerializeToString(Ptr<const AttributeChecker> checker) const          \
    {                                                                                              \
        std::ostringstream oss;                                                                    \
        oss << m_value;                                                                            \
        return oss.str();                                                                          \
    }                                                                                              \
    bool name##Value::DeserializeFromString(std::string value,                                     \
                                            Ptr<const AttributeChecker> checker)                   \
    {                                                                                              \
        if (value.empty())                                                                         \
        {                                                                                          \
            m_value = type();                                                                      \
            return true;                                                                           \
        }                                                                                          \
        std::istringstream iss;                                                                    \
        iss.str(value);                                                                            \
        iss >> m_value;                                                                            \
        NS_ABORT_MSG_UNLESS(iss.eof(),                                                             \
                            "Attribute value \"" << value << "\" is not properly formatted");      \
        return !iss.bad() && !iss.fail();                                                          \
    }

/**
 * @ingroup attributehelper
 *
 * Define the class methods belonging to
 * attribute value class \pname{typeValue} for class \pname{type}.
 *
 * @param [in] type The name of the class.
 *
 * This macro implements the \pname{typeValue} class methods
 * (including the \pname{typeValue}%::%SerializeToString
 * and \pname{typeValue}%::%DeserializeFromString methods).
 *
 * Typically invoked in the source file.
 */
#define ATTRIBUTE_VALUE_IMPLEMENT(type) ATTRIBUTE_VALUE_IMPLEMENT_WITH_NAME(type, type)

/**
 * @ingroup attributehelper
 *
 * Define the \c MaketypeChecker function for class \pname{type}.
 *
 * @param [in] type The name of the class
 *
 * This macro implements the \c MaketypeChecker function.
 *
 * Typically invoked in the source file..
 */
#define ATTRIBUTE_CHECKER_IMPLEMENT(type)                                                          \
    Ptr<const AttributeChecker> Make##type##Checker()                                              \
    {                                                                                              \
        return MakeSimpleAttributeChecker<type##Value, type##Checker>(#type "Value", #type);       \
    }

/**
 * @ingroup attributehelper
 *
 * Define the \c MaketypeChecker function for class \pname{type}.
 *
 * @param [in] type The name of the class.
 * @param [in] name The string name of the underlying type.
 *
 * This macro implements the \c MaketypeChecker function
 * for class \pname{type}.
 *
 * Typically invoked in the source file..
 */
#define ATTRIBUTE_CHECKER_IMPLEMENT_WITH_NAME(type, name)                                          \
    Ptr<const AttributeChecker> Make##type##Checker()                                              \
    {                                                                                              \
        return MakeSimpleAttributeChecker<type##Value, type##Checker>(#type "Value", name);        \
    }

/**
 * @ingroup attributehelper
 *
 * Declare the attribute value, accessor and checkers for class \pname{type}
 *
 * @param [in] type The name of the class
 *
 * This macro declares:
 *
 *   - The attribute value class \pname{typeValue},
 *
 *   - The attribute accessor functions \c MaketypeAccessor,
 *
 *   - The AttributeChecker class \pname{typeChecker},
 *     and the \c MaketypeChecker function,
 *
 * for class \pname{type}.
 *
 * This macro should be invoked outside of the class
 * declaration in its public header.
 */
#define ATTRIBUTE_HELPER_HEADER(type)                                                              \
    ATTRIBUTE_VALUE_DEFINE(type);                                                                  \
    ATTRIBUTE_ACCESSOR_DEFINE(type);                                                               \
    ATTRIBUTE_CHECKER_DEFINE(type)

/**
 * @ingroup attributehelper
 *
 * Define the attribute value, accessor and checkers for class \pname{type}
 *
 * @param [in] type The name of the class
 *
 * This macro implements
 *
 *   - The \pname{typeValue} class methods,
 *
 *   - The \c MaketypeChecker function,
 *
 * for class \pname{type}.
 *
 * This macro should be invoked from the class implementation file.
 */
#define ATTRIBUTE_HELPER_CPP(type)                                                                 \
    ATTRIBUTE_CHECKER_IMPLEMENT(type);                                                             \
    ATTRIBUTE_VALUE_IMPLEMENT(type)

#endif /* ATTRIBUTE_HELPER_H */
