# ruff: noqa: F401
import logging

from graphene import (
    ID,
    Boolean,
    Field,
    Int,
    List,
    Mutation,
    ObjectType,
    Schema,
    String,
)
from graphene_sqlalchemy import SQLAlchemyObjectType

from .base import CustomBase
from .models import Block, Grid, Model
from .setup import init

logger = logging.getLogger("uvicorn")

alchemy_to_graphene = dict(
    String="String",
    Boolean="Boolean",
    Integer="Int",
)


def generate_graphql_schema_code(models: list[type[CustomBase]]):
    graphql_types = ""
    query_c = """
class Query(ObjectType):
    """
    mutation_c = """
class Mutation(ObjectType):
    """
    code = ""

    for model in models:
        # names
        real_name = model.__name__
        one_s = real_name.lower()
        multiple_s = real_name.lower() + "s"

        # Types
        alchemy_name = f"{real_name}Type"
        graphql_types += f"""
class {alchemy_name}(SQLAlchemyObjectType):
    class Meta:
        model = {real_name}
        """

        # Queries
        query_c += f"""
    {one_s} = Field({real_name}Type, id=ID())
    {multiple_s} = Field(List({real_name}Type), limit=ID())
    def resolve_{one_s}(self, info, id):
        return {alchemy_name}.get_query(info).get(id)
    
    def resolve_{multiple_s}(self, info, limit=None):
        return {alchemy_name}.get_query(info).all()
        """

        # Gather fields
        args = {}
        for k, v in model.__table__.columns.items():
            if k == "id":
                continue

            alchemy_type_s = v.type.__class__.__name__
            graphene_type_s = alchemy_to_graphene[alchemy_type_s]
            args[k] = graphene_type_s

        fields_s = "\n".join([f"        {k} = {v}()" for k, v in args.items()])
        # Mutations
        mutation_name = f"{real_name}Mutation"
        graphql_types += f"""        
class {mutation_name}(Mutation):
    {one_s} = Field({alchemy_name})

    class Arguments:
        id = ID(required=True)
        delete = Boolean()
{fields_s}

    Output = {alchemy_name}

    def mutate(self, info, **params):
        print(params)
        res = None
        id = params.pop("id")
        delete = params.pop("delete", False)

        if id is not None and delete:
            raise ValueError(f"Cannot delete and update {{id=}} at the same time")

        if id is None:
            # Create a new item
            item = {real_name}(**params).save()
            res = {mutation_name}(item)
        else:
            # Update an existing item
            item = {real_name}.get(id)
            item = item.update(**params)
            res = {mutation_name}({one_s}=item)

        return res.{one_s}
        """

        mutation_c += f"""
    {one_s} = {real_name}Mutation.Field()
        """

    code += graphql_types
    code += query_c
    code += mutation_c
    return code


# class CreateBlock(Mutation):
#     block = Field(Block)

#     class Arguments:
#         name = String(required=True)

#     def mutate(self, info, name):
#         block = BlockDB(name=name).save()
#         return CreateBlock(block=block)


# class CreateGrid(Mutation):
#     grid = Field(Grid)

#     class Arguments:
#         name = String(required=True)

#     def mutate(self, info, name):
#         grid = GridDB(name=name).save()
#         return CreateGrid(grid=grid)


# class CreateModel(Mutation):
#     model = Field(Model)

#     class Arguments:
#         name = String(required=True)

#     def mutate(self, info, name):
#         model = ModelDB(name=name).save()
#         return CreateModel(model=model)


# class UpdateModel(Mutation):
#     model = Field(Model)

#     class Arguments:
#         id = ID(required=True)
#         name = String()

#     def mutate(self, info, id, name):
#         model = ModelDB.get(id).update(name=name).commit()
#         return UpdateModel(model=model)


# class DeleteModel(Mutation):
#     success = Field(lambda: String)

#     class Arguments:
#         id = ID(required=True)

#     def mutate(self, info, id):
#         ModelDB.get(id).delete()
#         return DeleteModel(success=f"Model {id} deleted successfully")

# class Mutations(ObjectType):
#     create_model = CreateModel.Field()
#     update_model = UpdateModel.Field()
#     delete_model = DeleteModel.Field()

#     create_block = CreateBlock.Field()


# Graphene doesn't allow easy creation / resolving of objects
# This here we do something really hacky to create the schema
code = generate_graphql_schema_code([Grid, Model, Block])
code += "\n"
print(code)
exec(code, globals(), locals())


schema = Schema(query=globals()["Query"], mutation=globals()["Mutation"])
init()
