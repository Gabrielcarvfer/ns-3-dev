/*
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Faker Moatamri <faker.moatamri@sophia.inria.fr>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "attribute-iterator.h"

#include <gtk/gtk.h>

namespace ns3
{

enum
{
    COL_NODE = 0,
    COL_LAST
};

/**
 * @ingroup configstore
 * @brief A class used in the implementation of the GtkConfigStore
 */
struct ModelNode
{
    /**
     * @brief node type structure
     */
    enum
    {
        // store object + attribute name
        NODE_ATTRIBUTE,
        // store object + attribute name
        NODE_POINTER,
        // store object + attribute name
        NODE_VECTOR,
        // store index + value (object)
        NODE_VECTOR_ITEM,
        // store object
        NODE_OBJECT
    } type; ///< node type

    std::string name;   ///< node name
    Ptr<Object> object; ///< the object
    uint32_t index;     ///< index
};

/**
 * @ingroup configstore
 * @brief ModelCreator class
 *
 */
class ModelCreator : public AttributeIterator
{
  public:
    ModelCreator();

    /**
     * Allocate attribute tree
     * @param treestore GtkTreeStore *
     */
    void Build(GtkTreeStore* treestore);

  private:
    void DoVisitAttribute(Ptr<Object> object, std::string name) override;
    void DoStartVisitObject(Ptr<Object> object) override;
    void DoEndVisitObject() override;
    void DoStartVisitPointerAttribute(Ptr<Object> object,
                                      std::string name,
                                      Ptr<Object> value) override;
    void DoEndVisitPointerAttribute() override;
    void DoStartVisitArrayAttribute(Ptr<Object> object,
                                    std::string name,
                                    const ObjectPtrContainerValue& vector) override;
    void DoEndVisitArrayAttribute() override;
    void DoStartVisitArrayItem(const ObjectPtrContainerValue& vector,
                               uint32_t index,
                               Ptr<Object> item) override;
    void DoEndVisitArrayItem() override;
    /**
     * Add item to attribute tree
     * @param node The model node
     */
    void Add(ModelNode* node);
    /// Remove current tree item
    void Remove();

    GtkTreeStore* m_treestore;         ///< attribute tree
    std::vector<GtkTreeIter*> m_iters; ///< attribute tree item
};
} // namespace ns3
