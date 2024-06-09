import { gql, useQuery } from "@apollo/client";

const GET_MODELS = gql`
  query {
    models {
      id
      name
    }
  }
`;

const Dashboard = () => {
  const { loading, error, data } = useQuery(GET_MODELS);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error : {error.message}</p>;

  const models = data.models;

  return (
    <div>
      <h1>Dashboard</h1>
      {models.map(({ name }: any) => {
        return <div>{name}</div>;
      })}
    </div>
  );
};

export default Dashboard;
