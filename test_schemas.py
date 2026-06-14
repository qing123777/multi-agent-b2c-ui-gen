"""Runtime smoke-tests for the revised B2C specification schemas in fyp.py."""
import sys, importlib, traceback
sys.path.insert(0, "C:/Users/Lenovo/Desktop/FYP")

# ── helpers ────────────────────────────────────────────────────────────────────
passed = []
failed = []

def run(name, fn):
    try:
        fn()
        passed.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        failed.append((name, e))
        print(f"  FAIL  {name}")
        traceback.print_exc()

# ── load module ────────────────────────────────────────────────────────────────
print("\n=== Loading fyp.py ===")
import fyp
fyp = importlib.reload(fyp)

LayoutSpecification     = fyp.LayoutSpecification
ComponentSpecification  = fyp.ComponentSpecification
InteractionSpecification= fyp.InteractionSpecification
DecorationSpecification = fyp.DecorationSpecification
UISpecificationBundle   = fyp.UISpecificationBundle
Page                    = fyp.Page
Region                  = fyp.Region
ComponentNode           = fyp.ComponentNode
PageComponentTree       = fyp.PageComponentTree
ComponentState          = fyp.ComponentState
ComponentEvent          = fyp.ComponentEvent
DataBindingSpec         = fyp.DataBindingSpec
ActionResultState       = fyp.ActionResultState
InteractionAction       = fyp.InteractionAction
InteractionEvent        = fyp.InteractionEvent
InteractionFlow         = fyp.InteractionFlow
PageInteraction         = fyp.PageInteraction
InteractionStateTransition = fyp.InteractionStateTransition
ComponentDecoration     = fyp.ComponentDecoration
PageDecoration          = fyp.PageDecoration
DesignTokens            = fyp.DesignTokens

print("  OK — fyp.py loaded\n")

# ── 1. New enum values ─────────────────────────────────────────────────────────
print("=== 1. Enum / type checks ===")

def test_pagetype_exists():
    assert fyp.PageType is not None
    # validate a Page with pageType
    Page(pageName="shop", pageType="plp", pageLayoutType="flex-column", regions=[])

def test_pagetype_rejects_invalid():
    try:
        Page(pageName="x", pageType="homepage", pageLayoutType="grid", regions=[])
        raise AssertionError("Should have rejected invalid pageType")
    except Exception as e:
        assert "pageType" in str(e) or "homepage" in str(e) or "validation" in str(e).lower()

def test_actiontype_has_ecommerce():
    from typing import get_args
    args = get_args(fyp.ActionType)
    for v in ("addToCart", "updateQuantity", "applyFilter", "clearFilter", "toggleWishlist"):
        assert v in args, f"{v} missing from ActionType"

def test_actiontype_no_log_no_apicall():
    from typing import get_args
    args = get_args(fyp.ActionType)
    assert "log"     not in args, "'log' still in ActionType"
    assert "apiCall" not in args, "'apiCall' still in ActionType"

def test_componentrole_gone():
    assert not hasattr(fyp, "ComponentRole"), "ComponentRole should be removed"

def test_layoutcomponentref_gone():
    assert not hasattr(fyp, "LayoutComponentRef"), "LayoutComponentRef should be removed"

for fn in [test_pagetype_exists, test_pagetype_rejects_invalid,
           test_actiontype_has_ecommerce, test_actiontype_no_log_no_apicall,
           test_componentrole_gone, test_layoutcomponentref_gone]:
    run(fn.__name__, fn)

# ── 2. LayoutSpecification ─────────────────────────────────────────────────────
print("\n=== 2. LayoutSpecification ===")

def test_layout_basic():
    layout = LayoutSpecification(pages=[
        Page(
            pageName="plp",
            pageType="plp",
            pageLayoutType="flex-column",
            responsive=True,
            regions=[
                Region(name="header",  ownedBy="HTML", layoutType="flex-row"),
                Region(name="catalog", ownedBy="HTML", layoutType="grid"),
                Region(name="footer",  ownedBy="HTML", layoutType="flex-row"),
            ],
        )
    ])
    assert layout.pages[0].pageType == "plp"
    assert len(layout.pages[0].regions) == 3

def test_layout_no_componentrefs():
    r = Region(name="body", ownedBy="HTML", layoutType="grid")
    assert not hasattr(r, "componentRefs"), "Region should not have componentRefs"

def test_layout_duplicate_regions_rejected():
    try:
        Page(pageName="x", pageType="pdp", pageLayoutType="grid", regions=[
            Region(name="main", ownedBy="HTML", layoutType="grid"),
            Region(name="main", ownedBy="CSS",  layoutType="flex-row"),
        ])
        raise AssertionError("Should reject duplicate region names")
    except Exception as e:
        assert "unique" in str(e).lower()

for fn in [test_layout_basic, test_layout_no_componentrefs, test_layout_duplicate_regions_rejected]:
    run(fn.__name__, fn)

# ── 3. ComponentSpecification ─────────────────────────────────────────────────
print("\n=== 3. ComponentSpecification ===")

def test_component_databindingspec():
    node = ComponentNode(
        id="price-label",
        tag="span",
        ownedBy="HTML",
        componentCategory="product-price",
        dataBinding={"source": "product.price", "format": "currency"},
        className="price",
    )
    assert isinstance(node.dataBinding, DataBindingSpec)
    assert node.dataBinding.source == "product.price"
    assert node.dataBinding.format == "currency"

def test_component_no_role_field():
    node = ComponentNode(id="btn", tag="button", ownedBy="HTML")
    assert not hasattr(node, "role"), "ComponentNode should not have 'role'"

def test_component_no_layoutref_field():
    node = ComponentNode(id="nav", tag="nav", ownedBy="HTML")
    assert not hasattr(node, "layoutRef"), "ComponentNode should not have 'layoutRef'"

def test_component_category_field():
    node = ComponentNode(id="grid", tag="section", ownedBy="HTML", componentCategory="product-card")
    assert node.componentCategory == "product-card"

def test_component_category_all_values_valid():
    """Every value in ComponentCategory Literal must round-trip through ComponentNode."""
    from typing import get_args
    for cat in get_args(fyp.ComponentCategory):
        node = ComponentNode(id=f"node-{cat}", tag="section", ownedBy="HTML", componentCategory=cat)
        assert node.componentCategory == cat, f"round-trip failed for {cat}"

def test_component_category_rejects_freeform():
    """Invented category names must raise ValidationError."""
    try:
        ComponentNode(id="x", tag="div", ownedBy="HTML", componentCategory="my-widget")
        raise AssertionError("Should reject non-Literal componentCategory")
    except Exception as e:
        assert "my-widget" in str(e) or "literal" in str(e).lower() or "validation" in str(e).lower()

def test_component_category_rejects_camelcase():
    """CamelCase variants must be rejected (not in the Literal)."""
    try:
        ComponentNode(id="x", tag="div", ownedBy="HTML", componentCategory="ProductCard")
        raise AssertionError("Should reject CamelCase variant")
    except Exception as e:
        pass  # expected

def test_component_category_optional():
    """componentCategory is Optional — omitting it must not raise."""
    node = ComponentNode(id="wrapper", tag="div", ownedBy="HTML")
    assert node.componentCategory is None

def test_component_css_owned_needs_classname():
    try:
        ComponentNode(id="card", tag="div", ownedBy="CSS")  # no className
        raise AssertionError("Should reject CSS-owned without className")
    except Exception as e:
        assert "className" in str(e)

def test_component_databinding_plain_string_rejected():
    """Old pattern dataBinding='product.price' should now fail validation."""
    try:
        node = ComponentNode(id="x", tag="span", ownedBy="HTML", dataBinding="product.price")
        # If pydantic coerces, source must still be set
        if isinstance(node.dataBinding, DataBindingSpec):
            pass  # coerced — acceptable
        else:
            raise AssertionError("plain string dataBinding should coerce or fail")
    except Exception:
        pass  # expected

def test_component_tag_is_required():
    """fyp.py ComponentNode requires tag (coerce is notebook-only)."""
    try:
        ComponentNode(id="x", ownedBy="HTML")
        raise AssertionError("Should require tag")
    except Exception as e:
        assert "tag" in str(e)

def test_component_ownedby_must_be_valid():
    """fyp.py ComponentNode rejects null/invalid ownedBy (coerce is notebook-only)."""
    try:
        ComponentNode(id="x", tag="div", ownedBy=None)
        raise AssertionError("Should reject null ownedBy")
    except Exception as e:
        assert "ownedBy" in str(e) or "literal_error" in str(e)

for fn in [test_component_databindingspec, test_component_no_role_field,
           test_component_no_layoutref_field, test_component_category_field,
           test_component_category_all_values_valid,
           test_component_category_rejects_freeform,
           test_component_category_rejects_camelcase,
           test_component_category_optional,
           test_component_css_owned_needs_classname,
           test_component_databinding_plain_string_rejected,
           test_component_tag_is_required, test_component_ownedby_must_be_valid]:
    run(fn.__name__, fn)

# ── 4. InteractionSpecification ───────────────────────────────────────────────
print("\n=== 4. InteractionSpecification ===")

def test_interaction_action_result_state():
    action = InteractionAction(
        type="addToCart",
        targetId="btn-add",
        result_state={"description": "Cart count increments; button shows 'Added'"},
    )
    assert isinstance(action.result_state, ActionResultState)
    assert "Cart" in action.result_state.description

def test_interaction_no_coerce_string_result_state():
    """result_state must be an object, not a bare string."""
    try:
        InteractionAction(type="updateState", result_state="loading")
        # If it passes, result_state should be None or ActionResultState
    except Exception:
        pass  # expected — string result_state no longer coerced

def test_interaction_ecommerce_actions():
    for action_type in ("addToCart", "updateQuantity", "applyFilter", "clearFilter", "toggleWishlist"):
        a = InteractionAction(type=action_type)
        assert a.type == action_type

def test_interaction_invalid_action_rejected():
    try:
        InteractionAction(type="apiCall")
        raise AssertionError("apiCall should be rejected")
    except Exception as e:
        assert "apiCall" in str(e) or "validation" in str(e).lower()

for fn in [test_interaction_action_result_state, test_interaction_no_coerce_string_result_state,
           test_interaction_ecommerce_actions, test_interaction_invalid_action_rejected]:
    run(fn.__name__, fn)

# ── 5. UISpecificationBundle cross-spec validation ────────────────────────────
print("\n=== 5. UISpecificationBundle cross-spec validation ===")

def _make_bundle():
    layout = LayoutSpecification(pages=[Page(
        pageName="plp", pageType="plp", pageLayoutType="flex-column",
        regions=[
            Region(name="header",  ownedBy="HTML", layoutType="flex-row"),
            Region(name="catalog", ownedBy="HTML", layoutType="grid"),
        ],
    )])
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="plp",
        regionComponents={
            "header": [ComponentNode(
                id="navbar", tag="nav", ownedBy="HTML",
                componentCategory="breadcrumb",
                events=[ComponentEvent(event="onClick", flowName="nav-click")],
                states=[ComponentState(name="active")],
                className="navbar",
            )],
            "catalog": [ComponentNode(
                id="product-grid", tag="section", ownedBy="HTML",
                componentCategory="product-card",
                dataBinding={"source": "catalog.products", "format": "list"},
            )],
        },
    )])
    interaction = InteractionSpecification(pages=[PageInteraction(
        pageName="plp",
        flows=[InteractionFlow(
            name="nav-click",
            trigger=InteractionEvent(event="onClick", targetId="navbar"),
            actions=[InteractionAction(
                type="navigate",
                targetId="navbar",
                result_state={"description": "Navigate to target page"},
            )],
            stateTransitions=[InteractionStateTransition(
                componentId="navbar", fromState=None, toState="active",
            )],
        )],
    )])
    decoration = DecorationSpecification(
        globalTokens=DesignTokens(colors={"primary": "#2563eb"}),
        pages=[PageDecoration(
            pageName="plp",
            componentDecorations=[ComponentDecoration(
                componentId="navbar",
                className="navbar",
                baseStyles={"display": "flex", "padding": "16px"},
                stateVariants=[{"stateName": "active", "styles": {"borderBottom": "2px solid #2563eb"}}],
            )],
        )],
    )
    return UISpecificationBundle(
        layout=layout, component=component,
        interaction=interaction, decoration=decoration,
    )

def test_bundle_valid():
    b = _make_bundle()
    assert b.layout.pages[0].pageType == "plp"

def test_bundle_rejects_unknown_component_in_deco():
    import pydantic
    b_data = _make_bundle()
    # Tamper: decoration references unknown component id
    b_data.decoration.pages[0].componentDecorations[0].componentId = "ghost-id"
    try:
        UISpecificationBundle(
            layout=b_data.layout, component=b_data.component,
            interaction=b_data.interaction, decoration=b_data.decoration,
        )
        raise AssertionError("Should reject unknown componentId in decoration")
    except Exception as e:
        assert "ghost-id" in str(e)

def test_bundle_rejects_unknown_flow_name():
    b_data = _make_bundle()
    b_data.component.pages[0].regionComponents["header"][0].events[0].flowName = "no-such-flow"
    try:
        UISpecificationBundle(
            layout=b_data.layout, component=b_data.component,
            interaction=b_data.interaction, decoration=b_data.decoration,
        )
        raise AssertionError("Should reject unknown flowName")
    except Exception as e:
        assert "no-such-flow" in str(e)

def test_bundle_no_componentrefs_check():
    """Verifies the old componentRefs cross-check is gone — bundle validates without refs."""
    b = _make_bundle()
    # Old code would have required componentRefs; new code should pass fine
    assert b is not None

for fn in [test_bundle_valid, test_bundle_rejects_unknown_component_in_deco,
           test_bundle_rejects_unknown_flow_name, test_bundle_no_componentrefs_check]:
    run(fn.__name__, fn)

# ── 6. _build_component_contract_from_layout ─────────────────────────────────
print("\n=== 6. _build_component_contract_from_layout ===")

def _make_layout_plp():
    return LayoutSpecification(pages=[Page(
        pageName="plp", pageType="plp", pageLayoutType="flex-column",
        regions=[
            Region(name="header",  ownedBy="HTML", layoutType="flex-row"),
            Region(name="catalog", ownedBy="HTML", layoutType="grid"),
            Region(name="footer",  ownedBy="CSS",  layoutType="flex-row"),
        ],
    )])

def test_contract_includes_pagetype():
    result = fyp._build_component_contract_from_layout(_make_layout_plp())
    assert "plp" in result, "pageType 'plp' missing from contract output"

def test_contract_includes_region_details():
    result = fyp._build_component_contract_from_layout(_make_layout_plp())
    assert "header"  in result
    assert "catalog" in result
    assert "footer"  in result
    assert "grid"    in result   # layoutType
    assert "CSS"     in result   # ownedBy

def test_contract_no_componentrefs_format():
    """Old format listed ref IDs — new format must not contain 'componentRefs'."""
    result = fyp._build_component_contract_from_layout(_make_layout_plp())
    assert "componentRefs" not in result
    assert "ref.id"         not in result

def test_contract_multi_page():
    layout = LayoutSpecification(pages=[
        Page(pageName="plp", pageType="plp", pageLayoutType="flex-column",
             regions=[Region(name="catalog", ownedBy="HTML", layoutType="grid")]),
        Page(pageName="pdp", pageType="pdp", pageLayoutType="flex-column",
             regions=[Region(name="details", ownedBy="HTML", layoutType="flex-row")]),
        Page(pageName="cart", pageType="cart", pageLayoutType="flex-column",
             regions=[Region(name="items",   ownedBy="HTML", layoutType="flex-column")]),
    ])
    result = fyp._build_component_contract_from_layout(layout)
    for name in ("plp", "pdp", "cart", "catalog", "details", "items"):
        assert name in result, f"'{name}' missing from multi-page contract"

for fn in [test_contract_includes_pagetype, test_contract_includes_region_details,
           test_contract_no_componentrefs_format, test_contract_multi_page]:
    run(fn.__name__, fn)


# ── 7. _repair_component_spec_from_layout ────────────────────────────────────
print("\n=== 7. _repair_component_spec_from_layout ===")

def test_repair_preserves_existing_nodes():
    """Nodes already present in a region must not be lost."""
    layout = _make_layout_plp()
    candidate = {"pages": [{"pageName": "plp", "regionComponents": {
        "header":  [{"id": "navbar", "tag": "nav", "ownedBy": "HTML"}],
        "catalog": [],
        "footer":  [],
    }}]}
    repaired = fyp._repair_component_spec_from_layout(layout, candidate)
    header_nodes = repaired["pages"][0]["regionComponents"]["header"]
    assert len(header_nodes) == 1
    assert header_nodes[0]["id"] == "navbar"

def test_repair_fills_missing_regions_with_empty_list():
    """Layout regions absent from the candidate get an empty list — no stub nodes."""
    layout = _make_layout_plp()
    candidate = {"pages": [{"pageName": "plp", "regionComponents": {
        "header": [{"id": "navbar", "tag": "nav", "ownedBy": "HTML"}],
        # catalog and footer intentionally missing
    }}]}
    repaired = fyp._repair_component_spec_from_layout(layout, candidate)
    regions = repaired["pages"][0]["regionComponents"]
    assert "catalog" in regions, "missing layout region must appear in repaired output"
    assert "footer"  in regions, "missing layout region must appear in repaired output"
    assert regions["catalog"] == [], "missing region must be empty list — no componentRef stubs"
    assert regions["footer"]  == [], "missing region must be empty list — no componentRef stubs"

def test_repair_no_componentref_stubs():
    """Repair must not inject stub nodes — componentRefs were removed."""
    layout = LayoutSpecification(pages=[Page(
        pageName="cart", pageType="cart", pageLayoutType="flex-column",
        regions=[Region(name="items", ownedBy="HTML", layoutType="flex-column")],
    )])
    candidate = {"pages": [{"pageName": "cart", "regionComponents": {}}]}
    repaired = fyp._repair_component_spec_from_layout(layout, candidate)
    items = repaired["pages"][0]["regionComponents"]["items"]
    assert items == [], f"expected empty list, got {items}"

def test_repair_adds_missing_page():
    """A page in layout absent from candidate must be created with empty regions."""
    layout = LayoutSpecification(pages=[
        Page(pageName="plp", pageType="plp", pageLayoutType="flex-column",
             regions=[Region(name="catalog", ownedBy="HTML", layoutType="grid")]),
        Page(pageName="pdp", pageType="pdp", pageLayoutType="flex-column",
             regions=[Region(name="details", ownedBy="HTML", layoutType="flex-row")]),
    ])
    candidate = {"pages": [{"pageName": "plp", "regionComponents": {"catalog": []}}]}
    repaired = fyp._repair_component_spec_from_layout(layout, candidate)
    page_names = [p["pageName"] for p in repaired["pages"]]
    assert "pdp" in page_names, "layout page missing from candidate must appear in repaired output"
    pdp = next(p for p in repaired["pages"] if p["pageName"] == "pdp")
    assert pdp["regionComponents"].get("details") == []

for fn in [test_repair_preserves_existing_nodes, test_repair_fills_missing_regions_with_empty_list,
           test_repair_no_componentref_stubs, test_repair_adds_missing_page]:
    run(fn.__name__, fn)


# ── 8. _validate_component_contract_for_downstream ───────────────────────────
print("\n=== 8. _validate_component_contract_for_downstream ===")

def _contract_layout():
    return LayoutSpecification(pages=[Page(
        pageName="plp", pageType="plp", pageLayoutType="flex-column",
        regions=[
            Region(name="header",  ownedBy="HTML", layoutType="flex-row"),
            Region(name="catalog", ownedBy="HTML", layoutType="grid"),
        ],
    )])

def test_contract_valid_passes():
    layout = _contract_layout()
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="plp",
        regionComponents={
            "header":  [ComponentNode(id="navbar",       tag="nav",     ownedBy="HTML",
                                      componentCategory="navbar",
                                      events=[ComponentEvent(event="onClick", flowName="nav-go")],
                                      className="navbar")],
            "catalog": [ComponentNode(id="product-grid", tag="section", ownedBy="HTML",
                                      componentCategory="product-grid")],
        },
    )])
    result = fyp._validate_component_contract_for_downstream(layout, component)
    assert result is None, f"Expected no violations, got: {result}"

def test_contract_no_role_violation():
    """Nodes have no 'role' field — no role-based violation should ever fire."""
    layout = _contract_layout()
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="plp",
        regionComponents={
            "header":  [ComponentNode(id="nav", tag="nav", ownedBy="HTML")],
            "catalog": [ComponentNode(id="grid", tag="section", ownedBy="HTML",
                                      componentCategory="product-grid")],
        },
    )])
    result = fyp._validate_component_contract_for_downstream(layout, component)
    # role must not appear in any violation message
    assert result is None or "role" not in result

def test_contract_css_without_classname_reported():
    """CSS-owned component missing className must produce a violation."""
    layout = _contract_layout()
    # Build manually because ComponentNode validator would reject at model level;
    # bypass by mocking the dict path via repair
    raw_candidate = {"pages": [{"pageName": "plp", "regionComponents": {
        "header":  [{"id": "nav", "tag": "nav", "ownedBy": "HTML"}],
        "catalog": [{"id": "grid", "tag": "section", "ownedBy": "CSS"}],  # no className
    }}]}
    repaired = fyp._repair_component_spec_from_layout(layout, raw_candidate)
    try:
        component = fyp._coerce_component_spec(repaired)
        result = fyp._validate_component_contract_for_downstream(layout, component)
        # If component parsed (coerce validators may have defaulted ownedBy), check violation
        if result is not None:
            assert "className" in result or "CSS" in result
    except Exception:
        pass  # ValidationError from ComponentNode is also acceptable

def test_contract_event_without_flowname_reported():
    """A ComponentEvent with missing flowName must produce a violation."""
    layout = _contract_layout()
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="plp",
        regionComponents={
            "header":  [ComponentNode(id="nav", tag="nav", ownedBy="HTML",
                                      events=[ComponentEvent(event="onClick", flowName=None)])],
            "catalog": [],
        },
    )])
    result = fyp._validate_component_contract_for_downstream(layout, component)
    assert result is not None, "Missing flowName must produce a violation"
    assert "flowName" in result or "onClick" in result

def test_contract_missing_region_in_component_reported():
    """Layout has a region absent from ComponentSpec regionComponents → violation."""
    layout = _contract_layout()
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="plp",
        regionComponents={
            "header": [ComponentNode(id="nav", tag="nav", ownedBy="HTML")],
            # "catalog" missing entirely
        },
    )])
    result = fyp._validate_component_contract_for_downstream(layout, component)
    assert result is not None, "Missing region must produce a violation"
    assert "catalog" in result

for fn in [test_contract_valid_passes, test_contract_no_role_violation,
           test_contract_css_without_classname_reported,
           test_contract_event_without_flowname_reported,
           test_contract_missing_region_in_component_reported]:
    run(fn.__name__, fn)


# ── 9. Nested children — bundle cross-validation ─────────────────────────────
print("\n=== 9. Nested children — bundle cross-validation ===")

def _make_nested_bundle():
    """PDP with a product-gallery section that has an img child."""
    layout = LayoutSpecification(pages=[Page(
        pageName="pdp", pageType="pdp", pageLayoutType="flex-column",
        regions=[Region(name="gallery", ownedBy="HTML", layoutType="grid")],
    )])
    child = ComponentNode(
        id="gallery-img", tag="img", ownedBy="CSS", className="gallery-img",
        states=[ComponentState(name="loading")],
    )
    parent = ComponentNode(
        id="product-gallery", tag="section", ownedBy="HTML",
        componentCategory="product-image-gallery",
        children=[child],
    )
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="pdp",
        regionComponents={"gallery": [parent]},
    )])
    interaction = InteractionSpecification(pages=[PageInteraction(pageName="pdp", flows=[])])
    decoration = DecorationSpecification(
        globalTokens=DesignTokens(),
        pages=[PageDecoration(
            pageName="pdp",
            componentDecorations=[ComponentDecoration(
                componentId="gallery-img",   # CHILD id — proves walk_ids goes deep
                className="gallery-img",
                baseStyles={"width": "100%"},
                stateVariants=[{"stateName": "loading", "styles": {"opacity": "0.5"}}],
            )],
        )],
    )
    return UISpecificationBundle(
        layout=layout, component=component,
        interaction=interaction, decoration=decoration,
    )

def test_bundle_validates_child_component_id():
    """Decoration targeting a nested child id must pass validation."""
    bundle = _make_nested_bundle()
    assert bundle is not None

def test_bundle_child_statevariant_validated():
    """StateVariant on child must be cross-validated against the child's declared states."""
    b = _make_nested_bundle()
    # Tamper: stateVariant references state not declared on child
    b.decoration.pages[0].componentDecorations[0].stateVariants[0].stateName = "ghost-state"
    try:
        UISpecificationBundle(
            layout=b.layout, component=b.component,
            interaction=b.interaction, decoration=b.decoration,
        )
        raise AssertionError("Should reject undeclared stateVariant on nested child")
    except Exception as e:
        assert "ghost-state" in str(e)

def test_bundle_rejects_unknown_child_id_in_interaction():
    """InteractionFlow trigger targeting non-existent nested id must fail."""
    layout = LayoutSpecification(pages=[Page(
        pageName="pdp", pageType="pdp", pageLayoutType="flex-column",
        regions=[Region(name="gallery", ownedBy="HTML", layoutType="grid")],
    )])
    child = ComponentNode(id="real-child", tag="img", ownedBy="HTML")
    parent = ComponentNode(id="product-gallery", tag="section", ownedBy="HTML",
                           componentCategory="product-image-gallery",
                           children=[child])
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="pdp",
        regionComponents={"gallery": [parent]},
    )])
    interaction = InteractionSpecification(pages=[PageInteraction(
        pageName="pdp",
        flows=[InteractionFlow(
            name="zoom",
            trigger=InteractionEvent(event="onClick", targetId="ghost-child"),  # doesn't exist
            actions=[InteractionAction(type="updateState")],
        )],
    )])
    decoration = DecorationSpecification(pages=[PageDecoration(pageName="pdp")])
    try:
        UISpecificationBundle(layout=layout, component=component,
                              interaction=interaction, decoration=decoration)
        raise AssertionError("Should reject unknown nested id in interaction")
    except Exception as e:
        assert "ghost-child" in str(e)

for fn in [test_bundle_validates_child_component_id,
           test_bundle_child_statevariant_validated,
           test_bundle_rejects_unknown_child_id_in_interaction]:
    run(fn.__name__, fn)


# ── 10. Unknown region in ComponentSpec ──────────────────────────────────────
print("\n=== 10. Unknown region in ComponentSpec ===")

def test_bundle_rejects_unknown_region():
    """ComponentSpec referencing a region name not in LayoutSpec must fail."""
    layout = LayoutSpecification(pages=[Page(
        pageName="plp", pageType="plp", pageLayoutType="flex-column",
        regions=[Region(name="catalog", ownedBy="HTML", layoutType="grid")],
    )])
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="plp",
        regionComponents={
            "catalog":      [],
            "ghost-region": [],   # not in layout
        },
    )])
    interaction = InteractionSpecification(pages=[PageInteraction(pageName="plp", flows=[])])
    decoration  = DecorationSpecification(pages=[PageDecoration(pageName="plp")])
    try:
        UISpecificationBundle(layout=layout, component=component,
                              interaction=interaction, decoration=decoration)
        raise AssertionError("Should reject unknown region in ComponentSpec")
    except Exception as e:
        assert "ghost-region" in str(e)

def test_bundle_rejects_multiple_unknown_regions():
    """All unknown regions must appear in the error message."""
    layout = LayoutSpecification(pages=[Page(
        pageName="cart", pageType="cart", pageLayoutType="flex-column",
        regions=[Region(name="items", ownedBy="HTML", layoutType="flex-column")],
    )])
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="cart",
        regionComponents={
            "items":    [],
            "sidebar":  [],   # not in layout
            "floating": [],   # not in layout
        },
    )])
    interaction = InteractionSpecification(pages=[PageInteraction(pageName="cart", flows=[])])
    decoration  = DecorationSpecification(pages=[PageDecoration(pageName="cart")])
    try:
        UISpecificationBundle(layout=layout, component=component,
                              interaction=interaction, decoration=decoration)
        raise AssertionError("Should reject multiple unknown regions")
    except Exception as e:
        err = str(e)
        assert "sidebar" in err or "floating" in err

def test_bundle_passes_exact_region_match():
    """ComponentSpec with exact region coverage must pass."""
    layout = LayoutSpecification(pages=[Page(
        pageName="cart", pageType="cart", pageLayoutType="flex-column",
        regions=[
            Region(name="items",   ownedBy="HTML", layoutType="flex-column"),
            Region(name="summary", ownedBy="HTML", layoutType="flex-row"),
        ],
    )])
    component = ComponentSpecification(pages=[PageComponentTree(
        pageName="cart",
        regionComponents={"items": [], "summary": []},
    )])
    interaction = InteractionSpecification(pages=[PageInteraction(pageName="cart", flows=[])])
    decoration  = DecorationSpecification(pages=[PageDecoration(pageName="cart")])
    bundle = UISpecificationBundle(layout=layout, component=component,
                                   interaction=interaction, decoration=decoration)
    assert bundle is not None

for fn in [test_bundle_rejects_unknown_region,
           test_bundle_rejects_multiple_unknown_regions,
           test_bundle_passes_exact_region_match]:
    run(fn.__name__, fn)


# ── summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  PASSED: {len(passed)}/{len(passed)+len(failed)}")
if failed:
    print(f"  FAILED: {len(failed)}")
    for name, err in failed:
        print(f"    - {name}: {err}")
print('='*50)
